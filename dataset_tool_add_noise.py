import os
import glob
import fnmatch
import PIL.Image
import numpy as np


def filter_image_sizes(images):
    filtered = []
    for idx, fname in enumerate(images):
        if (idx % 10000) == 0:
            print('loading images', idx, '/', len(images))
        try:
            with PIL.Image.open(fname) as img:
                w = img.size[0]
                h = img.size[1]
                if (w > 512 or h > 512) or (w < 256 or h < 256):
                    continue
                filtered.append(fname)
        except:
            print('Could not load image', fname, 'skipping file..')
    return filtered

def add_train_noise(style, params, x, img_range=255.0):
    shape = x.shape
    if style == "gauss_fix":
        std = params[0]
        noise = np.random.randn(shape[0], shape[1], shape[2]) * std * img_range
        return np.clip(x + noise, 0, img_range).astype(np.uint8)
    elif style == "gauss_range":
        min_std, max_std = params
        std = np.random.rand(shape[0], shape[1], shape[2]) * (max_std - min_std) + min_std
        noise = np.random.randn(shape) * std
        return np.clip(x + noise, 0, img_range).astype(np.uint8)


def load_and_save(img_path, style, params):
    img_name = os.path.basename(img_path)
    img = PIL.Image.open(img_path)
    img = img.convert("RGB")
    img = add_train_noise(style, params, np.asarray(img))
    save_path = os.path.join(save_dir, img_name)
    img = PIL.Image.fromarray(img)
    img.save(save_path, quality=100, subsampling=0)




input_dir = "/home/tomheaven/实验/ILSVRC2012/ILSVRC2012_img_val"
style = 'gauss_fix'
params = [0.25]
save_dir = "./Imagenet_val_%s_%.2f" % (style, params[0])

images = []
pattern = os.path.join(input_dir, '**/*')
all_fnames = glob.glob(pattern, recursive=True)
for fname in all_fnames:
    # include only JPEG/jpg/png
    if fnmatch.fnmatch(fname, '*.JPEG') or fnmatch.fnmatch(
            fname, '*.jpg') or fnmatch.fnmatch(fname, '*.png'):
        images.append(fname)
images = sorted(images)

filtered = filter_image_sizes(images)
print(len(filtered))

os.makedirs(save_dir, exist_ok=True)
for idx, img_path in enumerate(filtered):
    if (idx % 1000) == 0:
        print('loading and saving images', idx, '/', len(filtered))
    load_and_save(img_path, style, params)
print(len(glob.glob(os.path.join(save_dir, "*.JPEG"))))
