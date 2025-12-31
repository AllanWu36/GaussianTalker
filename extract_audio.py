
import argparse
import os
import subprocess
import sys

def extract_audio(video_path, output_path=None, start_time=None, end_time=None):
    if not os.path.exists(video_path):
        print(f"[ERROR] Video file not found: {video_path}")
        sys.exit(1)

    if output_path is None:
        base_name = os.path.splitext(video_path)[0]
        output_path = f"{base_name}.wav"

    print(f"[INFO] Extracting audio from {video_path} to {output_path}...")
    if start_time:
        print(f"       Start time: {start_time}s")
    if end_time:
        print(f"       End time: {end_time}s")
    
    cmd = ["ffmpeg", "-y", "-i", video_path]
    
    if start_time:
        cmd.extend(["-ss", str(start_time)])
    
    if end_time:
        cmd.extend(["-to", str(end_time)])
        
    cmd.extend(["-vn", output_path])
    
    try:
        subprocess.run(cmd, check=True)
        print(f"[SUCCESS] Audio saved to {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] FFmpeg failed: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Extract WAV audio from MP4 video.")
    parser.add_argument("video", help="Path to the input MP4 video file")
    parser.add_argument("--output", "-o", help="Path to the output WAV file (optional, defaults to input filename.wav)")
    parser.add_argument("--start", "-s", type=float, help="Start time in seconds")
    parser.add_argument("--end", "-e", type=float, help="End time in seconds")
    
    args = parser.parse_args()
    
    extract_audio(args.video, args.output, args.start, args.end)

if __name__ == "__main__":
    main()
