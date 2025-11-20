import os
from PIL import Image
from tqdm import tqdm


def load_gif(gif_path):
    """
    Load a GIF file and return the frames as a list of PIL Image objects.
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


def cut_image(img):
    # only keep the top half of the image
    width, height = img.size
    new_height = height // 2
    new_img = img.crop((0, 0, width, new_height))
    return new_img

def save_gif(frames, output_path):
    """
    Save a list of PIL Image objects as a GIF file.
    """
    frames[0].save(output_path, save_all=True, append_images=frames[1:], loop=0, duration=100)


if __name__ == "__main__":
    gif_root = '.'
    for gif_name in os.listdir(gif_root):
        gif_path = os.path.join(gif_root, gif_name)
        if not gif_path.endswith('.gif'):
            continue
        frames = load_gif(gif_path)
        new_frames = [cut_image(frame) for frame in frames]
        output_path = os.path.join(gif_root, f'cut_{gif_name}')
        save_gif(new_frames, output_path)