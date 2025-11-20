import cv2
import os
from PIL import Image


def find_all_videos(directory):
    """
    Find all video files in the given directory and its subdirectories.

    Args:
        directory (str): The directory to search for video files.

    Returns:
        list: A list of paths to video files.
    """
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
    video_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if any(file.endswith(ext) for ext in video_extensions):
                video_files.append(os.path.join(root, file))
    return video_files


def convert_video_to_gif(video_path, output_path):
    """
    Convert a video file to a GIF file.

    Args:
        video_path (str): The path to the input video file.
        output_path (str): The path to the output GIF file.
    """
    # Read the video file
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    # Convert frames to PIL images
    pil_frames = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in frames]

    # Save as GIF
    pil_frames[0].save(output_path, save_all=True, append_images=pil_frames[1:], loop=0, duration=100)


if __name__ == "__main__":
    # Directory containing video files
    input_directory = "factor"
    # Directory to save GIF files
    output_directory = "factor"

    # Find all video files in the input directory
    video_files = find_all_videos(input_directory)

    # Convert each video file to GIF
    for video_file in video_files:
        print(video_file)
        # output_file = os.path.join(output_directory, os.path.splitext(os.path.basename(video_file))[0] + ".gif")
        output_file = video_file.replace(".mp4", ".gif").replace(".avi", ".gif").replace(".mov", ".gif").replace(".mkv", ".gif").replace(".flv", ".gif").replace(".wmv", ".gif")
        convert_video_to_gif(video_file, output_file)
        print(f"Converted {video_file} to {output_file}")
