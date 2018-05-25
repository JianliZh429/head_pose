import os

BASE_DIR = os.path.dirname(__file__)


def sample_image(file_name):
    return os.path.join(BASE_DIR, file_name)
