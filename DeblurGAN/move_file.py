import os

def list_files(directory):
	fileList = [f for f in sorted(os.listdir(directory))]
	# filePath = [os.path.join(directory, f) for f in os.listdir(directory)]
	return fileList

# source_folder = '../GOPRO_dataset/test'
# dest_folder = '../GOPRO_dataset/test_modified'
#
# folders = list_files(source_folder)
# folders = [os.path.join(data_folder, f) for f in folders]

import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
import cv2

im = np.zeros((20, 20, 3), np.uint8)
# im[5:-5, 5:-5, 0] = 1
# im[5:-5, 5:-5, 1] = 0.5
# im[5:-5, 5:-5, 2] = 0.8
# im = ndimage.distance_transform_bf(im)
im_noise = np.zeros((20, 20, 3), np.uint8)
for i in range(3):
	im_slice = im[:,:,i]



# im_noise = cv2.randn(im,(0),(99))

# im_med = ndimage.median_filter(im_noise, 3)
im_med = np.zeros((20, 20, 3), np.uint8)
for i in range(3):
	im_noise_slice = im_noise[:,:,i]
	im_med[:,:,i] = ndimage.median_filter(im_noise_slice, 3)

plt.figure(figsize=(16, 5))

# plt.subplot(141)
# plt.imshow(im, interpolation='nearest')
# plt.axis('off')
# plt.title('Original image', fontsize=20)
plt.subplot(142)
# plt.imshow(im_noise, interpolation='nearest', vmin=0, vmax=5)
plt.imshow(im_noise, interpolation='nearest')
plt.axis('off')
plt.title('Noisy image', fontsize=20)
plt.subplot(143)
plt.imshow(im_med, interpolation='nearest')
plt.axis('off')
plt.title('Median filter', fontsize=20)
# plt.subplot(144)
# plt.imshow(np.abs(im - im_med), cmap=plt.cm.hot, interpolation='nearest')
# plt.axis('off')
# plt.title('Error', fontsize=20)

#
# plt.subplots_adjust(wspace=0.02, hspace=0.02, top=0.9, bottom=0, left=0,
#                     right=1)

plt.show()