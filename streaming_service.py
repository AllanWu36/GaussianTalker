import sys
import os
import argparse
import numpy as np
import torch
import cv2
import base64
import time
import tensorflow.compat.v1 as tf
from scipy.io import wavfile
import io

# Ensure we can import from local directories
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scene import Scene
from gaussian_renderer import GaussianModel, render_from_batch
from utils.general_utils import safe_state
from arguments import ModelParams, PipelineParams, ModelHiddenParams, get_combined_args

from data_utils.deepspeech_features.deepspeech_features import (
    prepare_deepspeech_net, 
    pure_conv_audio_to_deepspeech
)

class StreamingTalker:
    def __init__(self, model_path, deepspeech_model_path=None):
        self.model_path = model_path
        
        # 1. Setup Arguments
        self.parser = argparse.ArgumentParser()
        self.model_params = ModelParams(self.parser, sentinel=True)
        self.pipeline_params = PipelineParams(self.parser)
        self.hidden_params = ModelHiddenParams(self.parser)
        
        # Mock sys.argv for get_combined_args if needed, 
        # or just construct args manually. 
        # get_combined_args relies on sys.argv[1:] to parse "cmdline" args.
        # We can inject model_path into the parser defaults or manipulate sys.argv temporary if we want to rely on that util.
        # But simpler: emulate the result of get_combined_args logic + config file.
        
        # Let's try to construct a dummy args namespace and load config manually if methods allow, 
        # but get_combined_args is tightly coupled with argparse. 
        # We will trick it by setting defaults and calling it with empty list.
        
        # Temporary override sys.argv to avoid conflicts with fastapi args
        old_argv = sys.argv
        sys.argv = [sys.argv[0], "--model_path", model_path, "--eval"]
        
        try:
            self.args = get_combined_args(self.parser)
        finally:
            sys.argv = old_argv

        # Force some args
        self.args.model_path = model_path
        self.args.iteration = -1 # Load latest
        self.args.quiet = True
        
        safe_state(self.args.quiet)
        
        # 2. Load Gaussian Model
        print(f"[INFO] Loading Gaussian Model from {model_path}")
        self.dataset = self.model_params.extract(self.args)
        self.hyperparams = self.hidden_params.extract(self.args)
        
        self.gaussians = GaussianModel(self.dataset.sh_degree, self.hyperparams)
        self.scene = Scene(self.dataset, self.gaussians, load_iteration=self.args.iteration, shuffle=False)
        self.gaussians.eval()
        
        self.test_cameras = self.scene.getTrainCameras() # Use train cameras if test not available?
        if len(self.scene.getTestCameras()) > 0:
            self.test_cameras = self.scene.getTestCameras()
            
        self.ref_cam_idx = 0
        self.ref_cam = self.test_cameras[0]
        
        self.pipeline = PipelineParams(self.parser).extract(self.args) # Re-extract pure object

        # 3. Setup DeepSpeech
        if deepspeech_model_path is None:
             # Default path or try to find it
             deepspeech_model_path = 'deepspeech-0.9.2-models.pbmm' # Placeholder
             # Check if exists, else look in home dir as per original script
             if not os.path.exists(deepspeech_model_path):
                 # Try typical paths
                 expanded = os.path.expanduser('~/.tensorflow/models/deepspeech-0_1_0-b90017e8.pb')
                 if os.path.exists(expanded):
                     deepspeech_model_path = expanded
                 else:
                     print("[WARN] DeepSpeech model not found. Please specify path.")

        print(f"[INFO] Loading DeepSpeech from {deepspeech_model_path}")
        self.ds_graph, self.ds_logits, self.ds_input, self.ds_lengths = prepare_deepspeech_net(deepspeech_model_path)
        self.ds_sess = tf.compat.v1.Session(graph=self.ds_graph)

        # 4. State
        self.audio_buffer = np.array([], dtype=np.int16)
        self.sample_rate = 16000 # Target SR
        self.processed_samples = 0
        
        # How many samples do we need for one video frame (at 25fps)?
        # 16000 / 25 = 640 samples per frame.
        self.samples_per_frame = 640
        self.frame_counter = 0
        
        # Context required for DeepSpeech
        # It needs some context. We'll use a sliding window logic.
        self.ds_hz = 50 
        self.video_fps = 25
        
    def process_chunk(self, audio_bytes):
        """
        Receives raw audio bytes (16k, mono, 16bit PCM).
        Returns list of JPEG encoded frames.
        """
        # 1. Convert bytes to numpy
        new_audio = np.frombuffer(audio_bytes, dtype=np.int16)
        self.audio_buffer = np.concatenate([self.audio_buffer, new_audio])
        
        generated_frames = []
        
        # 2. Check if we have enough data for new frames.
        # We need a context window of +/- 8 frames (0.32s) roughly for Gaussian Talker
        # Plus DeepSpeech context.
        # Let's say we process in chunks of X frames.
        
        # To simplify, we keep a "working" audio buffer that includes history.
        # We define a "valid" region we can render.
        # "att_mode=2" needs index +/- 4 features. 
        # DeepSpeech features are 50Hz (20ms). Video is 25Hz (40ms).
        # So 1 video frame = 2 DS features. 
        # index +/- 4 video frames = +/- 8 DS features.
        # So we need roughly 16 DS features of context?
        
        # Let's operate on units of "Video Frames".
        # If we have enough audio for (Processed + 1) + Context, we render Processed + 1.
        
        lookahead_frames = 8 # Future frames needed
        min_samples = (self.frame_counter + 1 + lookahead_frames) * self.samples_per_frame
        
        if len(self.audio_buffer) >= min_samples:
            # We can process frames.
            # But converting WHOLE buffer to DS features every time is slow.
            # Optimization: Just convert the relevant window.
            # Relevant window: [CurrentFrame - Context, CurrentFrame + Lookahead]
            
            # DeepSpeech window logic is complex. 
            # safe window: 1 sec before, 1 sec after.
            
            # Let's extract a safe window of audio
            center_sample = (self.frame_counter * self.samples_per_frame) - self.processed_samples
            start_sample = max(0, center_sample - 16000) # 1 sec history
            end_sample = len(self.audio_buffer) # As much future as possible
            
            chunk_audio = self.audio_buffer[start_sample:end_sample]
            
            # Run DeepSpeech on this chunk
            ds_features = self._run_deepspeech(chunk_audio, self.sample_rate)
            
            # ds_features is [Num_DS_Frames, 29]
            # We need to find the index corresponding to self.frame_counter
            # The chunk started at `start_sample`.
            # `center_sample` is relative to `start_sample`: `center_sample - start_sample`
            
            offset_seconds = (center_sample - start_sample) / self.sample_rate
            # DS features are 50Hz.
            ds_center_idx = int(offset_seconds * 50)
            
            # Currently just render ONE frame at a time if possible, or batch.
            # Let's check how many frames we can render.
            # We need ds_features up to ds_center_idx + 4 * (50/25) = +8?
            # Actually inference.py says:
            # att_mode=2: left = index - 4, right = index + 4 (indices of VIDEO frames)
            # The features passed to get_audio_features are Video-FPS aligned features?
            # extract_ds_features.py -> interpolate_features -> matches Video FPS.
            # So `aud_features` in inference.py are ALREADY 25FPS (or whatever args.fps is).
            
            # YES: extract_ds_features.py calls interpolate_features to output_rate=video_fps.
            # So if we map DS features to video FPS, we get a 1-to-1 correspondence.
            
            # So, re-plan:
            # 1. Get raw DS features (50Hz).
            # 2. Interpolate to 25Hz.
            # 3. Use index `frame_counter` (relative to the chunk start) to window.
            
            video_features = self._interpolate_features(ds_features, 50, self.video_fps)
            # video_features shape [N_Video_Frames, 29]
            
            video_center_idx = int(offset_seconds * self.video_fps)
            
            # Check if we have enough context for video_center_idx (need +/- 4)
            if video_center_idx + 4 < len(video_features):
                # We can render!
                # Extract the 8-frame window (+/-4)
                # But wait, inference.py `get_audio_features` for mode 2:
                # independent of FPS, it uses hardcoded indices +/- 4?
                # Yes: left=index-4, right=index+4.
                
                # So we take slice [idx-4 : idx+4].
                # If idx < 4, pad left.
                
                # Render
                window_feat = self._get_audio_window(video_features, video_center_idx)
                
                image = self._render_frame(window_feat, self.frame_counter)
                
                # Encode to JPEG
                _, buffer = cv2.imencode('.jpg', image)
                jpg_as_text = base64.b64encode(buffer).decode('utf-8')
                generated_frames.append(jpg_as_text)
                
                self.frame_counter += 1
                
                # Cleanup audio buffer
                # We need context for the NEXT frame.
                # Next frame needs audio at: (self.frame_counter) * self.samples_per_frame
                # Context is roughly 1 second history for safety.
                # So we can remove anything older than current frame time - 1 sec.
                
                required_history = 16000 * 1
                current_frame_sample_idx = self.frame_counter * self.samples_per_frame
                # But frame_counter is strictly increasing, while buffer is being trimmed?
                # No, if we trim buffer, we need to adjust frame_counter or track "buffer_start_offset".
                
                # Easier approach:
                # Reset audio_buffer and frame_counter periodically? No, that causes glitches.
                # Shift strategy:
                # 1. processed_samples measures absolute time.
                # 2. audio_buffer stores [samples_start_idx : ]
                # 3. When converting to DS, we calculate offset.
                
                # IMPLEMENTATION OF SHIFT (Simpler for this single iteration fix):
                # Just keep growing for demo purposes is risky.
                # Let's simple check: if buffer > 10 seconds?
                if len(self.audio_buffer) > 16000 * 10:
                     # Cut 5 seconds
                     cut_amt = 16000 * 5
                     self.audio_buffer = self.audio_buffer[cut_amt:]
                     # We must adjust frame_counter to match the new buffer?
                     # frame_counter counts frames since START of stream.
                     # We need to map frame_counter to index in audio_buffer.
                     # frame idx N corresponds to sample N * 640.
                     # If we cut first X samples, then index in buffer is (N * 640) - cut_amt.
                     self.samples_per_frame # is 640
                     
                     # To maintain sync without complicated offsets, let's keep it growing for the user session 
                     # but warn about it. For a proper fix we need a 'samples_consumed' offset.
                     
                     self.processed_samples += cut_amt # Reuse this var to track offset
                     
        # Ideally we loop to generate as many frames as possible
        # For now, return what we have
        return generated_frames
        
    # We need to adjust logic to use self.processed_samples as offset
    # In the code above:
    # center_sample = (self.frame_counter) * self.samples_per_frame
    # This assumes frame_counter starts at 0 and buffer includes ALL history.
    # If we trim buffer, we need:
    # center_sample_in_buffer = (self.frame_counter * self.samples_per_frame) - self.processed_samples


    def _run_deepspeech(self, audio, sr):
        # Wrapper for pure_conv_audio_to_deepspeech
        # We need to define the net_fn which uses self.ds_sess
        
        def net_fn(x):
            return self.ds_sess.run(
                self.ds_logits,
                feed_dict={
                    self.ds_input: x[np.newaxis, ...],
                    self.ds_lengths: [x.shape[0]]
                }
            )
            
        # Warning: pure_conv_audio_to_deepspeech interpolates internally!
        # But we want raw DS features first usually?
        # No, the function `pure_conv_audio_to_deepspeech` returns the WINDOWS directly if called as is.
        # But wait, looking at `pure_conv_audio_to_deepspeech` code:
        # It calls interpolate_features.
        # It calls `np.concatenate` to make windows.
        # It returns `np.array(windows)`.
        
        # We probably want the INTERMEDIATE output (just interpolated features), not the windowed ones, 
        # so we can manage windowing ourselves for streaming (feature continuity).
        # Or we use it as is, but it might be inefficient.
        
        # Let's modify behavior or copy the logic.
        # copying logic here is safer.
        
        # 1. Resample if needed
        target_sample_rate = 16000
        # Assuming input is already 16k
        
        # 2. Input vector
        # imports needed inside class method or global
        from data_utils.deepspeech_features.deepspeech_features import conv_audio_to_deepspeech_input_vector
        
        input_vector = conv_audio_to_deepspeech_input_vector(
            audio=audio.astype(np.int16),
            sample_rate=target_sample_rate,
            num_cepstrum=26,
            num_context=9
        )
        
        network_output = net_fn(input_vector)
        return network_output[:, 0] # Shape [Time, 29]

    def _interpolate_features(self, features, input_rate, output_rate):
        from data_utils.deepspeech_features.deepspeech_features import interpolate_features
        # We need to know output length.
        input_len = features.shape[0]
        input_seconds = input_len / input_rate
        output_len = int(round(input_seconds * output_rate))
        
        return interpolate_features(features, input_rate, output_rate, output_len)

    def _get_audio_window(self, features, index):
        # att_mode = 2 logic
        left = index - 4
        right = index + 4
        pad_left = 0
        pad_right = 0
        
        if left < 0:
            pad_left = -left
            left = 0
        
        if right > features.shape[0]:
            # This shouldn't happen if we checked bounds, but for safety
            pad_right = right - features.shape[0]
            right = features.shape[0]
            
        auds = features[left:right]
        
        if pad_left > 0:
            auds = np.concatenate([np.zeros((pad_left, *auds.shape[1:]), dtype=auds.dtype), auds], axis=0)
        if pad_right > 0:
            auds = np.concatenate([auds, np.zeros((pad_right, *auds.shape[1:]), dtype=auds.dtype)], axis=0)
            
        return torch.from_numpy(auds).float()

    def _render_frame(self, audio_window, frame_idx):
        # Clone camera
        import copy
        # Use ref cam based on frame_idx to loop?
        # inference.py: ref_idx = i % len(test_cameras)
        ref_idx = frame_idx % len(self.test_cameras)
        ref_cam = self.test_cameras[ref_idx]
        
        new_cam = copy.deepcopy(ref_cam)
        new_cam.aud_f = audio_window.cuda()
        new_cam.uid = frame_idx
        
        with torch.no_grad():
            output = render_from_batch(
                [new_cam],
                self.gaussians,
                self.pipeline,
                random_color=False,
                stage='fine',
                batch_size=1,
                visualize_attention=False,
                only_infer=True
            )
            img = output["rendered_image_tensor"][0]
            img = img.permute(1, 2, 0).detach().cpu().numpy()
            img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
            # RGB to BGR for opencv
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            return img

