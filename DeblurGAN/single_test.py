import time
import os
from DeblurGAN.options.test_options import TestOptions
from DeblurGAN.data.data_loader import CreateDataLoader
from DeblurGAN.models.models import create_model
from DeblurGAN.util.visualizer import Visualizer
from pdb import set_trace as st
from DeblurGAN.util import html
from DeblurGAN.util.metrics import PSNR
from ssim import SSIM
from PIL import Image
from torch.autograd import Variable
import matplotlib.pyplot as plt

# opt.dataroot', required=True, help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
# 		opt.batchSize', type=int, default=1, help='input batch size')
# 		opt.loadSizeX', type=int, default=640, help='scale images to this size')
# 		opt.loadSizeY', type=int, default=360, help='scale images to this size')
# 		opt.fineSize', type=int, default=256, help='then crop to this size')
# 		opt.input-nc', type=int, default=3, help='# of input image channels')
# 		opt.output-nc', type=int, default=3, help='# of output image channels')
# 		opt.ngf', type=int, default=64, help='# of gen filters in first conv layer')
# 		opt.ndf', type=int, default=64, help='# of discrim filters in first conv layer')
# 		opt.which-model-netD', type=str, default='basic', help='selects model to use for netD')
# 		opt.which-model-netG', type=str, default='resnet-9blocks', help='selects model to use for netG')
# 		opt.learn-residual', action='store-true', help='if specified, model would learn only the residual to the input')
# 		opt.gan-type', type=str, default='wgan-gp', help='wgan-gp : Wasserstein GAN with Gradient Penalty, lsgan : Least Sqaures GAN, gan : Vanilla GAN')
# 		opt.n-layers-D', type=int, default=3, help='only used if which-model-netD==n-layers')
# 		opt.gpu-ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
# 		opt.name', type=str, default='experiment-name', help='name of the experiment. It decides where to store samples and models')
# 		opt.dataset-mode', type=str, default='single', help='chooses how datasets are loaded. [unaligned | aligned | single]')
# 		opt.model', type=str, default='content-gan', help='chooses which model to use. pix2pix, test, content-gan')
# 		opt.which-direction', type=str, default='AtoB', help='AtoB or BtoA')
# 		opt.nThreads', default=2, type=int, help='# threads for loading data')
# 		opt.checkpoints-dir', type=str, default='./checkpoints', help='models are saved here')
# 		opt.norm', type=str, default='instance', help='instance normalization or batch normalization')
# 		opt.serial-batches', action='store-true', help='if true, takes images in order to make batches, otherwise takes them randomly')
# 		opt.display-winsize', type=int, default=256,  help='display window size')
# 		opt.display-id', type=int, default=1, help='window id of the web display')
# 		opt.display-port', type=int, default=8097, help='visdom port of the web display')
# 		opt.display-single-pane-ncols', type=int, default=0, help='if positive, display all images in a single visdom web panel with certain number of images per row.')
# 		opt.no-dropout', action='store-true', help='no dropout for the generator')
# 		opt.max-dataset-size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max-dataset-size, only a subset is loaded.')
# 		opt.resize-or-crop', type=str, default='resize-and-crop', help='scaling and cropping of images at load time [resize-and-crop|crop|scale-width|scale-width-and-crop]')
# 		opt.no-flip', action='store_true', help='if specified, do not flip the images for data augmentation')

opt = TestOptions().parse()
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()

model = create_model(opt)
for i, data in enumerate(dataset):
	input = data['A']
	model.set_input_direct(input) # data['A'] = FloatTensor (1, 3, 256, 256) [0, 1]
	model.test()
	plt.imshow((input[0, :, :, :] * 0.5 + 0.5).transpose(0, 2).transpose(0, 1).numpy())
	plt.show()
	visuals = model.get_current_visuals()
	output = visuals['fake_B']/255.0 # np.array (256, 256, 3) [0, 255]
	plt.imshow(output)
	plt.show()
	pass



# --dataroot ./dataset/blurred/part1/ --model test --dataset_mode single --learn_residual
