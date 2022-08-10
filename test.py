from __future__ import division
import os
import time
import glob
import datetime
import argparse
import numpy as np

import cv2
from PIL import Image
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from arch_unet import UNet

DEBUG = False

def load_checkpoint(net, opt):
    save_model_path = os.path.join(opt.save_model_path, opt.log_name + ".pth")
    net.load_state_dict(torch.load(save_model_path))
    print('Checkpoint loaded from {}'.format(save_model_path))
    return net

def get_generator():
    global operation_seed_counter
    operation_seed_counter += 1
    g_cuda_generator = torch.Generator(device="cuda")
    g_cuda_generator.manual_seed(operation_seed_counter)
    return g_cuda_generator


class AugmentNoise(object):
    def __init__(self, style):
        print(style)
        if style.startswith('gauss'):
            self.params = [
                float(p) / 255.0 for p in style.replace('gauss', '').split('_')
            ]
            if len(self.params) == 1:
                self.style = "gauss_fix"
            elif len(self.params) == 2:
                self.style = "gauss_range"
        elif style.startswith('poisson'):
            self.params = [
                float(p) for p in style.replace('poisson', '').split('_')
            ]
            if len(self.params) == 1:
                self.style = "poisson_fix"
            elif len(self.params) == 2:
                self.style = "poisson_range"

    def add_train_noise(self, x):
        shape = x.shape
        if self.style == "gauss_fix":
            std = self.params[0]
            std = std * torch.ones((shape[0], 1, 1, 1), device=x.device)
            noise = torch.cuda.FloatTensor(shape, device=x.device)
            torch.normal(mean=0.0,
                         std=std,
                         generator=get_generator(),
                         out=noise)
            return x + noise
        elif self.style == "gauss_range":
            min_std, max_std = self.params
            std = torch.rand(size=(shape[0], 1, 1, 1),
                             device=x.device) * (max_std - min_std) + min_std
            noise = torch.cuda.FloatTensor(shape, device=x.device)
            torch.normal(mean=0, std=std, generator=get_generator(), out=noise)
            return x + noise
        elif self.style == "poisson_fix":
            lam = self.params[0]
            lam = lam * torch.ones((shape[0], 1, 1, 1), device=x.device)
            noised = torch.poisson(lam * x, generator=get_generator()) / lam
            return noised
        elif self.style == "poisson_range":
            min_lam, max_lam = self.params
            lam = torch.rand(size=(shape[0], 1, 1, 1),
                             device=x.device) * (max_lam - min_lam) + min_lam
            noised = torch.poisson(lam * x, generator=get_generator()) / lam
            return noised

    def add_valid_noise(self, x):
        shape = x.shape
        if self.style == "gauss_fix":
            std = self.params[0]
            return np.array(x + np.random.normal(size=shape) * std,
                            dtype=np.float32)
        elif self.style == "gauss_range":
            min_std, max_std = self.params
            std = np.random.uniform(low=min_std, high=max_std, size=(1, 1, 1))
            return np.array(x + np.random.normal(size=shape) * std,
                            dtype=np.float32)
        elif self.style == "poisson_fix":
            lam = self.params[0]
            return np.array(np.random.poisson(lam * x) / lam, dtype=np.float32)
        elif self.style == "poisson_range":
            min_lam, max_lam = self.params
            lam = np.random.uniform(low=min_lam, high=max_lam, size=(1, 1, 1))
            return np.array(np.random.poisson(lam * x) / lam, dtype=np.float32)


def space_to_depth(x, block_size):
    n, c, h, w = x.size()
    unfolded_x = torch.nn.functional.unfold(x, block_size, stride=block_size)
    return unfolded_x.view(n, c * block_size**2, h // block_size,
                           w // block_size)

def validation_kodak(dataset_dir):
    fns = glob.glob(os.path.join(dataset_dir, "*"))
    fns.sort()
    images = []
    for fn in fns:
        im = Image.open(fn)
        im = np.array(im, dtype=np.float32)
        images.append(im)
    return images


def validation_bsd300(dataset_dir):
    fns = []
    fns.extend(glob.glob(os.path.join(dataset_dir, "test", "*")))
    fns.sort()
    images = []
    for fn in fns:
        im = Image.open(fn)
        im = np.array(im, dtype=np.float32)
        images.append(im)
    return images


def validation_Set14(dataset_dir):
    fns = glob.glob(os.path.join(dataset_dir, "*"))
    fns.sort()
    images = []
    for fn in fns:
        im = Image.open(fn)
        im = np.array(im, dtype=np.float32)
        images.append(im)
    return images


def ssim(prediction, target):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2
    img1 = prediction.astype(np.float64)
    img2 = target.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) *
                (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                       (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(target, ref):
    '''
    calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    img1 = np.array(target, dtype=np.float64)
    img2 = np.array(ref, dtype=np.float64)
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1[:, :, i], img2[:, :, i]))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


def calculate_psnr(target, ref):
    img1 = np.array(target, dtype=np.float32)
    img2 = np.array(ref, dtype=np.float32)
    diff = img1 - img2
    psnr = 10.0 * np.log10(255.0 * 255.0 / np.mean(np.square(diff)))
    return psnr

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--noisetype", type=str, default="gauss25")
    parser.add_argument('--val_dirs', type=str, default='./validation')
    parser.add_argument('--save_model_path', type=str, default='./pretrained_model')
    parser.add_argument('--log_name', type=str, default='model_gauss25_b4e100r02')
    parser.add_argument('--gpu_devices', default='3', type=str)
    parser.add_argument('--parallel', action='store_true')
    parser.add_argument('--n_feature', type=int, default=48)
    parser.add_argument('--n_channel', type=int, default=3)

    opt, _ = parser.parse_known_args()
    systime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
    operation_seed_counter = 0
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_devices
    # Validation Set
    Kodak_dir = os.path.join(opt.val_dirs, "Kodak")
    BSD300_dir = os.path.join(opt.val_dirs, "BSD300")
    Set14_dir = os.path.join(opt.val_dirs, "Set14")
    valid_dict = {
        "Kodak": validation_kodak(Kodak_dir),
        "BSD300": validation_bsd300(BSD300_dir),
        "Set14": validation_Set14(Set14_dir)
    }

    # Noise adder
    noise_adder = AugmentNoise(style=opt.noisetype)

    # Network
    network = UNet(in_nc=opt.n_channel,
                out_nc=opt.n_channel,
                n_feature=opt.n_feature)
    if opt.parallel:
        network = torch.nn.DataParallel(network)
    network = load_checkpoint(network, opt)
    network = network.cuda()
    print('init finish')
    network.eval()
    # validation
    save_model_path = os.path.join(opt.save_model_path, opt.log_name, systime)
    validation_path = os.path.join(save_model_path, "validation")
    os.makedirs(validation_path, exist_ok=True)
    np.random.seed(101)
    #valid_repeat_times = {"Kodak": 10, "BSD300": 3, "Set14": 20}
    valid_repeat_times = {"Kodak": 1, "BSD300": 1, "Set14": 1}

    for valid_name, valid_images in valid_dict.items():
        psnr_result = []
        ssim_result = []
        repeat_times = valid_repeat_times[valid_name]
        for i in range(repeat_times):
            for idx, im in enumerate(valid_images):
                origin255 = im.copy()
                origin255 = origin255.astype(np.uint8)
                im = np.array(im, dtype=np.float32) / 255.0
                noisy_im = noise_adder.add_valid_noise(im)
                if True:
                    noisy255 = noisy_im.copy()
                    noisy255 = np.clip(noisy255 * 255.0 + 0.5, 0,
                                    255).astype(np.uint8)
                # padding to square
                H = noisy_im.shape[0]
                W = noisy_im.shape[1]
                val_size = (max(H, W) + 31) // 32 * 32
                noisy_im = np.pad(
                    noisy_im,
                    [[0, val_size - H], [0, val_size - W], [0, 0]],
                    'reflect')
                transformer = transforms.Compose([transforms.ToTensor()])
                noisy_im = transformer(noisy_im)
                noisy_im = torch.unsqueeze(noisy_im, 0)
                noisy_im = noisy_im.cuda()
                with torch.no_grad():
                    if DEBUG:
                        print('noisy_im', noisy_im.shape)
                    prediction = network(noisy_im)
                    prediction = prediction[:, :, :H, :W]
                prediction = prediction.permute(0, 2, 3, 1)
                prediction = prediction.cpu().data.clamp(0, 1).numpy()
                prediction = prediction.squeeze()
                pred255 = np.clip(prediction * 255.0 + 0.5, 0,
                                255).astype(np.uint8)
                # calculate psnr
                cur_psnr = calculate_psnr(origin255.astype(np.float32),
                                        pred255.astype(np.float32))
                psnr_result.append(cur_psnr)
                cur_ssim = calculate_ssim(origin255.astype(np.float32),
                                        pred255.astype(np.float32))
                ssim_result.append(cur_ssim)

                # visualization
                if True:
                    save_path = os.path.join(
                        validation_path,
                        "{}_{:03d}_clean.png".format(
                            valid_name, idx))
                    Image.fromarray(origin255).convert('RGB').save(
                        save_path)
                    save_path = os.path.join(
                        validation_path,
                        "{}_{:03d}_noisy.png".format(
                            valid_name, idx))
                    Image.fromarray(noisy255).convert('RGB').save(
                        save_path)
                if i == 0:
                    save_path = os.path.join(
                        validation_path,
                        "{}_{:03d}_denoised.png".format(
                            valid_name, idx))
                    Image.fromarray(pred255).convert('RGB').save(save_path)

        psnr_result = np.array(psnr_result)
        avg_psnr = np.mean(psnr_result)
        avg_ssim = np.mean(ssim_result)
        log_path = os.path.join(validation_path,
                                "A_log_{}.csv".format(valid_name))
        with open(log_path, "a") as f:
            f.writelines("{},{}\n".format(avg_psnr, avg_ssim))
    
            
