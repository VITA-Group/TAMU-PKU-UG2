import os
import sys
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from PIL import Image
from keras import backend as KTF
from keras.preprocessing import image
from DeblurGAN.main import deblurGAN

parser = argparse.ArgumentParser()
parser.add_argument("--in_path", default="sample/evaluation-images/", help="input image/directory path")
parser.add_argument("--out_path", default="sample/evaluation-images_deblurGAN/", help="output image/directory path")
args = parser.parse_args()

input_path = args.in_path
output_path = args.out_path


def processImg(img_in, img_out):
    img = image.load_img(img_in)
    img = image.img_to_array(img)
    img = deblurGAN(img)
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
