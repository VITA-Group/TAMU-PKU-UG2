import os
import sys
import shutil
from tqdm import tqdm
from PIL import Image
from skimage import exposure
from keras.preprocessing import image

# To run:
# Format:	python HistEq/process.py path/to/input/ path/to/output
# Example:	python HistEq/process.py sample/evaluation-images sample/evaluation-images-histeq

input_path = os.path.abspath(sys.argv[1])
output_path = os.path.abspath(sys.argv[2])


def histogram_equalization(img):
    img = img / 255.
    img = exposure.equalize_adapthist(img)
    img = img * 255.
    return img


def processImg(img_in, img_out):
    img = image.load_img(img_in)
    img = image.img_to_array(img)
    img = histogram_equalization(img)
    img = Image.fromarray(img.astype('uint8'))
    img.save(img_out)


def processDir(dir_in, dir_out):
    if not os.path.exists(dir_out):
        os.makedirs(dir_out)

    for img in os.listdir(dir_in):
        processImg(os.path.join(dir_in, img), os.path.join(dir_out, img))


print("input_path: ", input_path)
print("output_path: ", output_path)

if input_path.endswith("jpg") or input_path.endswith("png"):
    processImg(input_path, output_path)
else:
    processDir(input_path, output_path)
