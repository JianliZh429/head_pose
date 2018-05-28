import glob
import os

BASE_DIR = os.path.dirname(__file__)


def sample_image(file_name):
    return os.path.join(BASE_DIR, file_name)


def sample_images(images_dir):
    files_extend = ['jpg', 'jpeg', 'gif', 'png']
    files = []
    for ext in files_extend:
        files += glob.glob('{}/*.{}'.format(images_dir, ext))
    return files
