import os
import platform

__author__ = 'roeiherz'

FILE_EXISTS_ERROR = (17, 'File exists')

IMG_FOLDER = 'images'
ANNOTATION_FOLDER = 'annotations'
DEBUG_MODE = False # 'ubuntu' not in os.environ['HOME']
if DEBUG_MODE:
    IMG_FOLDER += '_debug'
    ANNOTATION_FOLDER += '_debug'


def create_folder(path):
    """
    Checks if the path exists, if not creates it.
    :param path: A valid path that might not exist
    :return: An indication if the folder was created
    """
    folder_missing = not os.path.exists(path)

    if folder_missing:
        # Using makedirs since the path hierarchy might not fully exist.
        try:
            os.makedirs(path)
        except OSError as e:
            if (e.errno, e.strerror) == FILE_EXISTS_ERROR:
                print(e)
            else:
                raise

        print('Created folder {0}'.format(path))

    return folder_missing


def root_dir():
    if platform.system() == 'Linux':
        return os.path.join(os.getenv('HOME'), 'Documents', 'SKU110K')
    elif platform.system() == 'Windows':
        return os.path.abspath('C:/Users/{}/Documents/SKU110K/'.format(os.getenv('username')))


def image_path():
    return os.path.join(root_dir(), IMG_FOLDER)


def annotation_path():
    return os.path.join(root_dir(), ANNOTATION_FOLDER)
