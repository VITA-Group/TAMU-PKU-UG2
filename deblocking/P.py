
# coding: utf-8

# In[1]:

import os
import torch
import argparse
from torch import nn
from torch.autograd import Variable as var
from PIL import Image as im
from PIL import Image
from PIL import JpegPresets
from skimage.measure import compare_psnr, compare_ssim
import time
import math
import random
import numpy as np
from numpy import array as na
import matplotlib.pyplot as plt
import torchvision
import torch.optim as optim
import sys
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--weights', '-w', default='./P.weights')
parser.add_argument('--cpu', action='store_true')
parser.add_argument('--input_dir', '-i', required=True)
parser.add_argument('--output_dir', '-o', required=True)
args = parser.parse_args()

device = torch.device('cpu' if args.cpu else 'cuda')

def ensure_exists(dname):
    import os
    if not os.path.exists(dname):
        try:
            os.makedirs(dname)
        except:
            pass
    return dname

class JPEGImageDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None, preload=False, channel=-1):
        self.root_dir = root_dir
        self.transform = transform
        self.image_list = []
        self.name_list = []
        self.channel = channel

        for r, d, filenames in os.walk(self.root_dir):
            for f in filenames:
                if f[-3:] not in ['jpg', 'png']:
                    continue
                self.image_list.append(os.path.join(r, f))
                self.name_list.append(f)

        self.loaded_images = [None] * len(self.image_list)

        if preload:
            for idx in tqdm(range(len(self.image_list))):
                tmp = plt.imread(self.image_list[idx])
                if tmp.dtype == np.float32:
                    tmp = np.asarray(tmp*255, dtype=np.uint8)
                if len(tmp.shape) == 3:
                    tmp = tmp[:,:,:3]
                if self.channel != -1:
                    tmp = tmp[:,:,self.channel]
                self.loaded_images[idx] = tmp

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        if self.loaded_images[idx] is None:
            tmp = plt.imread(self.image_list[idx])
            if tmp.dtype == np.float32:
                tmp = np.asarray(tmp*255, dtype=np.uint8)
            if len(tmp.shape) == 3:
                tmp = tmp[:,:,:3]
            if self.channel != -1:
                tmp = tmp[:,:,self.channel]
            self.loaded_images[idx] = tmp
        ret = self.loaded_images[idx][:]
        if self.transform:
            ret = self.transform(ret)
        return ret

    def getName(self, idx):
        return self.name_list[idx]
   
class Align2(object):
    def __init__(self, l):
        self.l = l

    def __call__(self, image):
        h, w = image.shape[:2]
        new_h, new_w = self.l * (h // self.l), self.l * (w // self.l)

        image = image[:new_h, :new_w]
        return image

class JointNet3(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.conv0 = nn.Sequential(
            nn.Conv2d(C, 16, 3, 1, 1),
            nn.PReLU(init=0.1)
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.PReLU(init=0.1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.PReLU(init=0.1)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.PReLU(init=0.1)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.PReLU(init=0.1)
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.PReLU(init=0.1)
        )
        self.diconv1 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 2, dilation = 2),
            nn.PReLU(init=0.1)
        )
        self.diconv2 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 4, dilation = 4),
            nn.PReLU(init=0.1)
        )
        self.diconv3 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 8, dilation = 8),
            nn.PReLU(init=0.1)
        )
        self.conv8 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.PReLU(init=0.1)
        )
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReflectionPad2d((1, 0, 1, 0)),
            nn.AvgPool2d(2, stride = 1),
            nn.PReLU(init=0.1)
        )
        self.conv9 = nn.Sequential(
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.PReLU(init=0.1)
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReflectionPad2d((1, 0, 1, 0)),
            nn.AvgPool2d(2, stride = 1),
            nn.PReLU(init=0.1)
        )
        self.conv10 = nn.Sequential(
            nn.Conv2d(64, 16, 3, 1, 1),
            nn.PReLU(init=0.1)
        )
        self.output = nn.Sequential(
            nn.Conv2d(16, C, 5, 1, 2)
        )

    def forward(self, inputs):
        x = inputs
                
        x = self.conv0(x)
        x = self.conv1(x)
        res1 = x
        x = self.conv2(x)
        x = self.conv3(x)
        res2 = x
        x = self.conv4(x)
        x = self.conv6(x)
        x = self.diconv1(x)
        x = self.diconv2(x)
        x = self.diconv3(x)
        x = self.conv8(x)
        x = self.deconv1(x)
        x = torch.cat([x, res2], dim=1)
        x = self.conv9(x)
        x = self.deconv2(x)
        x = torch.cat([x, res1], dim=1)
        x = self.conv10(x)
        x = self.output(x) + inputs
        
        return x

def output2na(out):
    p = out.data.cpu().numpy()[0]
    p = p.transpose((1, 2, 0))
    p = 255 * p
    p[p > 255] = 255
    p[p < 0] = 0
    p = np.uint8(p)
    return p

# In[ ]:

jnet = nn.DataParallel(JointNet3(3))
jnet.load_state_dict(torch.load(args.weights))
jnet = jnet.to(device)


# In[ ]:
print("loading dataset...")
testset = JPEGImageDataset(args.input_dir, preload=True,
                           transform=torchvision.transforms.Compose([
                               Align2(24),
                               torchvision.transforms.ToTensor()
                           ]))
testset_loader = torch.utils.data.DataLoader(testset, batch_size=1)
_ps=[]

print("running...")
with torch.no_grad():
    for _, vd in tqdm(enumerate(testset_loader, 0)):
        gen = var(vd).float().to(device)
        bs, c, h, w = gen.shape

        hs, ws  = h // 3, w // 3
        out = jnet(gen[:,:,:hs,:ws])
        p1 = output2na(out)
        out = jnet(gen[:,:,hs:2*hs,:ws])
        p2 = output2na(out)
        out = jnet(gen[:,:,2*hs:,:ws])
        p3 = output2na(out)
        out = jnet(gen[:,:,:hs,ws:2*ws])
        p4 = output2na(out)
        out = jnet(gen[:,:,hs:2*hs,ws:2*ws])
        p5 = output2na(out)
        out = jnet(gen[:,:,2*hs:,ws:2*ws])
        p6 = output2na(out)
        out = jnet(gen[:,:,:hs,2*ws:])
        p7 = output2na(out)
        out = jnet(gen[:,:,hs:2*hs,2*ws:])
        p8 = output2na(out)
        out = jnet(gen[:,:,2*hs:,2*ws:])
        p9 = output2na(out)
        out = np.hstack([
            np.vstack([p1,p2,p3]),
            np.vstack([p4,p5,p6]),
            np.vstack([p7,p8,p9])
        ])
        result = testset.loaded_images[_].copy()
        result[:out.shape[0],:out.shape[1]] = out
        _ps.append(result)

print("writing...")
ensure_exists(args.output_dir)
for i in tqdm(range(len(testset_loader))):
    plt.imsave(os.path.join(args.output_dir, testset.getName(i)[:-3] + 'png'), _ps[i])

