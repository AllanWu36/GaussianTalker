
import argparse
import os
import sys
import subprocess
import torch
import numpy as np
import imageio
import cv2
from tqdm import tqdm
from argparse import Namespace
import importlib 
import shutil

try:
    from scene import Scene
    from gaussian_renderer import GaussianModel, render_from_batch
    from utils.general_utils import safe_state
    from arguments import ModelParams, PipelineParams, ModelHiddenParams, get_combined_args
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

def convert_audio(input_path, output_path):
    print(f"[INFO] Converting audio to 16k mono: {input_path} -> {output_path}")
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-ac", "1", "-ar", "16000",
        output_path
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def extract_audio_features(wav_path, npy_path):
    print(f"[INFO] Extracting DeepSpeech features...")
    # Call the existing script
    script_path = "data_utils/deepspeech_features/extract_ds_features.py"
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"Could not find feature extraction script at {script_path}")
    
    cmd = [
        "python", script_path,
        "--input", wav_path,
        "--output", npy_path
    ]
    subprocess.run(cmd, check=True)

def write_video(frames, audio_path, output_path, fps=25):
    print(f"[INFO] Writing video to {output_path}...")
    temp_vid = output_path + ".temp.mp4"
    imageio.mimwrite(temp_vid, frames, fps=fps, quality=8, output_params=['-vf', f'fps={fps}'], macro_block_size=None)
    
    cmd = [
        "ffmpeg", "-y",
        "-i", temp_vid,
        "-i", audio_path,
        "-c:v", "copy",
        "-c:a", "aac",
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-shortest",
        output_path
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if os.path.exists(temp_vid):
        os.remove(temp_vid)

def get_audio_features(features, att_mode, index):
    if att_mode == 0:
        return features[[index]]
    elif att_mode == 1:
        left = index - 8
        pad_left = 0
        if left < 0:
            pad_left = -left
            left = 0
        auds = features[left:index]
        if pad_left > 0:
            auds = torch.cat([torch.zeros(pad_left, *auds.shape[1:], device=auds.device, dtype=auds.dtype), auds], dim=0)
        return auds
    elif att_mode == 2:
        left = index - 4
        right = index + 4
        pad_left = 0
        pad_right = 0
        if left < 0:
            pad_left = -left
            left = 0
        if right > features.shape[0]:
            pad_right = right - features.shape[0]
            right = features.shape[0]
        auds = features[left:right]
        if pad_left > 0:
            auds = torch.cat([torch.zeros((pad_left, *auds.shape[1:]), device=auds.device, dtype=auds.dtype), auds], dim=0)
        if pad_right > 0:
            auds = torch.cat([auds, torch.zeros((pad_right, *auds.shape[1:]), device=auds.device, dtype=auds.dtype)], dim=0)
        return auds
    else:
        raise NotImplementedError(f'wrong att_mode: {att_mode}')

def main():
    parser = argparse.ArgumentParser()
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    hyperparam = ModelHiddenParams(parser)
    
    parser.add_argument("--wav", type=str, required=True, help="Path to input audio wav file")
    parser.add_argument("--output", type=str, default="output.mp4", help="Path to output video file")
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--configs", type=str)

    args = get_combined_args(parser)
    
    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)

    safe_state(args.quiet)
    
    # 1. Prepare Audio
    temp_dir = "temp_inference"
    os.makedirs(temp_dir, exist_ok=True)
    
    base_name = os.path.basename(args.wav).split('.')[0]
    processed_wav = os.path.join(temp_dir, f"{base_name}_16k.wav")
    processed_npy = os.path.join(temp_dir, f"{base_name}_16k.npy")
    
    try:
        convert_audio(args.wav, processed_wav)
        extract_audio_features(processed_wav, processed_npy)
        
        # Load features
        aud_features = np.load(processed_npy)
        aud_features = torch.from_numpy(aud_features).float()
        
        # Handle shape [N, 16, 29] -> [N, 29, 16] if necessary
        # The logic in talking_dataset_readers checks for 3 dims.
        if len(aud_features.shape) == 3:
             aud_features = aud_features.permute(0, 2, 1)
        
        # 2. Load Model
        print(f"[INFO] Loading Model from {args.model_path}")
        dataset = model.extract(args)
        hyperparams = hyperparam.extract(args)
        
        gaussians = GaussianModel(dataset.sh_degree, hyperparams)
        scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
        
        gaussians.eval()
        
        # 3. Prepare Cameras
        # We need to determine the number of frames based on audio features.
        # DeepSpeech features usually align with 25FPS if processed correctly?
        # Actually extract_ds_features produces features. We assume 1 feature vector = 1 video frame.
        total_frames = aud_features.shape[0]
        
        print(f"[INFO] Audio has {total_frames} frames. Preparing camera sequence...")
        
        test_cameras = scene.getTestCameras()
        if len(test_cameras) == 0:
            print("[WARN] No test cameras found, using train cameras as reference.")
            test_cameras = scene.getTrainCameras()
            
        custom_cams = []
        for i in range(total_frames):
            ref_idx = i % len(test_cameras)
            ref_cam = test_cameras[ref_idx]
            
            # Get specific audio window
            # Note: We need to move audio to appropriate device
            current_aud = get_audio_features(aud_features, att_mode=2, index=i)
            # current_aud shape should be [window, dim] e.g. [8, 16]
            
            # We create a new camera by copying important attributes.
            # But the Camera class is complex.
            # We can just assign the audio to the ref_cam?
            # Problem: If we modify ref_cam.aud_f, and we use the same ref_cam instance multiple times in a batch?
            # Creating a shallow copy of the object wrapper might be safer if Camera is mutable.
            # Or just modifying it if we process sequentially.
            # Since we will loop and render one by one or batch by batch...
            # The safer way is to clone the camera object using its copy method if available, or copy module.
            
            # Check if Camera has copy method. Yes it does in utils.py.
            # new_cam = ref_cam.copy() 
            import copy
            new_cam = copy.deepcopy(ref_cam)
            new_cam.aud_f = current_aud.cuda() # Ensure it's on GPU
            new_cam.uid = i # Update UID just in case
            
            custom_cams.append(new_cam)
            
        # 4. Render
        print(f"[INFO] Rendering {len(custom_cams)} frames...")
        images = []
        
        # Batch size for rendering? 
        # Using 1 for simplicity and safety with custom modified cameras.
        for i, cam in tqdm(enumerate(custom_cams), total=len(custom_cams)):
            with torch.no_grad():
                output = render_from_batch(
                    [cam], 
                    gaussians, 
                    pipeline, 
                    random_color=False, 
                    stage='fine', 
                    batch_size=1, 
                    visualize_attention=False, 
                    only_infer=True
                )
                img = output["rendered_image_tensor"][0]
                img = img.permute(1, 2, 0).detach().cpu().numpy()
                img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
                images.append(img)
        
        # 5. Save
        write_video(images, processed_wav, args.output, fps=args.fps)
        print(f"[SUCCESS] Saved video to {args.output}")

    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

if __name__ == "__main__":
    main()
