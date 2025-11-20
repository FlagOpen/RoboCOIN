from PIL import Image
from tqdm import tqdm


def load_gif(gif_path):
    """
    Load a GIF file and return a list of frames.
    """
    with Image.open(gif_path) as img:
        frames = []
        try:
            while True:
                frames.append(img.copy())
                img.seek(img.tell() + 1)
        except EOFError:
            pass
    return frames


def resize_gif(gif_path, save_path, size=None, factor=None):
    assert (size is not None) ^ (factor is not None), "Either size or factor must be provided, but not both."

    frames = load_gif(gif_path)
    
    if size is not None:
        new_size = size
    elif factor is not None:
        original_size = frames[0].size
        new_size = (int(original_size[0] * factor), int(original_size[1] * factor))
    else:
        raise ValueError("Either size or factor must be provided.")
    
    resized_frames = [frame.resize(new_size, Image.LANCZOS) for frame in frames]
    resized_frames[0].save(
        save_path,
        save_all=True,
        append_images=resized_frames[1:],
        loop=0,
        duration=frames[0].info.get('duration', 100),
        optimize=True
    )


def find_gifs(directory):
    """
    Find all GIF files in the specified directory.
    """
    import os
    gifs = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.gif'):
                gifs.append(os.path.join(root, file))
    return gifs


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Resize GIFs in a directory.")
    parser.add_argument("directory", help="Directory containing GIF files.")
    parser.add_argument("--size", type=int, nargs=2, metavar=('WIDTH', 'HEIGHT'), help="New size for the GIFs.")
    parser.add_argument("--factor", type=float, help="Resize factor for the GIFs.")

    args = parser.parse_args()

    gifs = find_gifs(args.directory)
    for gif in tqdm(gifs):
        save_path = gif.replace('.gif', '_thumb.gif')
        resize_gif(gif, save_path, size=args.size, factor=args.factor)
        print(f"Resized {gif} and saved to {save_path}.")
