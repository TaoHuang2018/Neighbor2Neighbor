import os
import glob
import fnmatch
import PIL.Image


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


def load_and_save(img_path):
    img_name = os.path.basename(img_path)
    img = PIL.Image.open(img_path)
    img = img.convert("RGB")
    save_path = os.path.join(save_dir, img_name)
    img.save(save_path, quality=100, subsampling=0)


input_dir = "./ILSVRC2012_img_val"
save_dir = "./Imagenet_val"

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
    load_and_save(img_path)
print(len(glob.glob(os.path.join(save_dir, "*.JPEG"))))
