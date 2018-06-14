import os
import time
import torch
import numpy as np
from DeblurGAN.options.test_options import TestOptions
from DeblurGAN.models.models import create_model


def deblurGAN(input):
    opt = TestOptions().parse()
    opt.nThreads = 1  # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip
    model = create_model(opt)
    input = input / 255.0
    # plt.imshow(input) # input numpy array (256, 256, 3)
    # plt.show()
    width = input.shape[0]
    height = input.shape[1]
    if width % 4 != 0 or height % 4 != 0:
        width_res = 4 - (width % 4)
        height_res = 4 - (height % 4)
        width_pad_left = np.round(width_res / 2).astype(int)
        height_pad_left = np.round(height_res / 2).astype(int)
        width_pad_right = width_res - width_pad_left
        height_pad_right = height_res - height_pad_left
        # input = np.pad(input, ((width_pad_left, width_pad_right), (height_pad_left, height_pad_right), (0,0)), mode='constant')
        input = np.pad(input, ((width_pad_left, width_pad_right), (height_pad_left, height_pad_right), (0, 0)),
                       mode='reflect')
    input = torch.from_numpy((input - 0.5) / 0.5).transpose(0, 2).transpose(1, 2).expand(1, -1, -1, -1)
    model.set_input_direct(input)  # data['A'] = FloatTensor (1, 3, 256, 256) [0, 1]
    model.test()
    visuals = model.get_current_visuals()
    output = visuals['fake_B']  # np.array (256, 256, 3) [0, 255]
    if width % 4 != 0 or height % 4 != 0:
        output = output[width_pad_left:-width_pad_right, height_pad_left:-height_pad_right, :]
    # output = output/255.0
    # plt.imshow(output)
    # plt.show()

    return output.astype(np.float32)
