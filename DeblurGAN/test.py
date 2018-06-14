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

opt = TestOptions().parse()
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
model = create_model(opt)
visualizer = Visualizer(opt)
# create website
web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
# test
avgPSNR = 0.0
avgSSIM = 0.0
counter = 0

for i, data in enumerate(dataset):
	if i >= opt.how_many:
		break
	counter = i
	model.set_input(data)
	model.test()
	visuals = model.get_current_visuals()
	# avgPSNR += PSNR(visuals['fake_B'],visuals['real_B'])
	# modified 03/04
	# psnr = PSNR(visuals['fake_B'],visuals['real_A'])
	psnr = PSNR(visuals['Restored_Train'],visuals['Sharp_Train'])
	avgPSNR += psnr
	# pilFake = Image.fromarray(visuals['fake_B'])
	# pilReal = Image.fromarray(visuals['real_A'])
	pilFake = Image.fromarray(visuals['Restored_Train'])
	pilReal = Image.fromarray(visuals['Sharp_Train'])
	# avgSSIM += SSIM(pilFake).cw_ssim_value(pilReal)
	# modified 03/04
	ssim= SSIM(pilFake).cw_ssim_value(pilReal)
	avgSSIM += ssim
	img_path = model.get_image_paths()
	print('process image... %s' % img_path)
	visualizer.save_images(webpage, visuals, img_path, psnr, ssim)
	
avgPSNR /= counter
avgSSIM /= counter
print('PSNR = %f, SSIM = %f' %
				  (avgPSNR, avgSSIM))

webpage.save()
