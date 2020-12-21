# -*- coding: utf-8 -*-
import random
from PIL import Image, ImageOps, ImageFilter, ImageFile
import numpy as np
import os
import torch
import torch.utils.data as data
import torchvision.transforms as ttransforms
import imageio
import time
import glob
import sys


ImageFile.LOAD_TRUNCATED_IMAGES = True

DEBUG = False

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
NUM_CLASSES = 19

# colour map
label_colours_19 = [
        # [  0,   0,   0],
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [0, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32],
        [  0,   0,   0]] # the color of ignored label(-1) 
label_colours_19 = list(map(tuple, label_colours_19))


# [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 15, 17, 18]
# colour map
label_colours_16 = [
        # [  0,   0,   0],
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [0, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 60, 100],
        [0, 0, 230],
        [119, 11, 32],
        [  0,   0,   0]] # the color of ignored label(-1)
label_colours_16 = list(map(tuple, label_colours_16))





class City_Dataset(data.Dataset):
    def __init__(self,
                 args,
                 data_root_path=os.path.abspath('./datasets/Cityscapes'),
                 list_path=os.path.abspath('./datasets/Cityscapes'),
                 split='train',
                 base_size=769,
                 crop_size=769,
                 training=True,
                 class_16=False,
                 class_13=False):

        self.args = args
        self.data_path=data_root_path
        self.list_path=list_path
        self.split=split
        if DEBUG: print('DEBUG: Cityscapes {0:} dataset path is {1:}'.format(self.split, self.list_path))
        self.base_size=base_size
        self.crop_size=crop_size
        if DEBUG: print('DEBUG: Cityscapes {0:} dataset image size is {1:}'.format(self.split, self.crop_size))

        self.base_size = self.base_size if isinstance(self.base_size, tuple) else (self.base_size, self.base_size)
        self.crop_size = self.crop_size if isinstance(self.crop_size, tuple) else (self.crop_size, self.crop_size)
        self.training = training

        self.random_mirror = args.random_mirror
        self.random_crop = args.random_crop
        self.resize = args.resize
        self.gaussian_blur = args.gaussian_blur

        ###
        item_list_filepath = os.path.join(self.list_path, self.split + ".txt")
        try:
            time.sleep(2)
            self.items = [id for id in open(item_list_filepath)]
        except FileNotFoundError:
            print('sys.argv', os.path.dirname(sys.argv[0]))
            print('FileNotFoundError: cwdir is ' + os.getcwd())
            print(os.listdir(os.getcwd()))
            print('FileNotFoundError: parent cwdir is ' + os.path.dirname(os.getcwd()))
            print('glob:', glob.glob(os.path.dirname(os.getcwd())+ '/*'))
            print('os:', os.listdir(os.path.dirname(os.getcwd())))
            print('FileNotFoundError: parent of parent cwdir is ' + os.path.dirname(os.path.dirname(os.getcwd())))
            print(os.listdir(os.path.dirname(os.path.dirname(os.getcwd()))))
            self.items = [id for id in open(item_list_filepath)]
        ###

        ignore_label = -1

        ###
        # self.id_to_trainid = {i:i for i in range(-1,19)}
        # self.id_to_trainid[255] = ignore_label

        self.id_to_trainid = {-1: ignore_label, 0: ignore_label, 1: ignore_label, 2: ignore_label,
                              3: ignore_label, 4: ignore_label, 5: ignore_label, 6: ignore_label,
                              7: 0, 8: 1, 9: ignore_label, 10: ignore_label, 11: 2, 12: 3, 13: 4,
                              14: ignore_label, 15: ignore_label, 16: ignore_label, 17: 5,
                              18: ignore_label, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
                              28: 15, 29: ignore_label, 30: ignore_label, 31: 16, 32: 17, 33: 18}
        ###


        # In SYNTHIA-to-Cityscapes case, only consider 16 shared classes
        self.class_16 = class_16
        ###
        synthia_set_16 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 15, 17, 18]
        self.trainid_to_16id = {id:i for i,id in enumerate(synthia_set_16)}
        self.trainid_to_16id[255] = ignore_label
        ###

        self.class_13 = class_13
        synthia_set_13 = [0, 1, 2, 6, 7, 8, 10, 11, 12, 13, 15, 17, 18]
        self.trainid_to_13id = {id:i for i,id in enumerate(synthia_set_13)}

        if DEBUG: print('DEBUG: Cityscapes {0:} -> item_list_filepath: {1:} , first item: {2:}'.format(self.split, item_list_filepath, self.items[0]))
        if DEBUG: print("{} num images in Cityscapes {} set have been loaded.".format(len(self.items), self.split))
        if self.args.numpy_transform:
            if DEBUG: print("used numpy_transform, instead of tensor transform!")

    def id2trainId(self, label, reverse=False, ignore_label=-1):
        label_copy = ignore_label * np.ones(label.shape, dtype=np.float32)
        for k, v in self.id_to_trainid.items():
            label_copy[label == k] = v
        if self.class_16:
            label_copy_16 = ignore_label * np.ones(label.shape, dtype=np.float32)
            for k, v in self.trainid_to_16id.items():
                label_copy_16[label_copy == k] = v
            label_copy = label_copy_16
        if self.class_13:
            label_copy_13 = ignore_label * np.ones(label.shape, dtype=np.float32)
            for k, v in self.trainid_to_13id.items():
                label_copy_13[label_copy == k] = v
            label_copy = label_copy_13
        return label_copy

    def __getitem__(self, item):
        ###
        id_img, id_gt = self.items[item].strip('\n').split(' ')
        image_path = self.data_path + id_img
        image = Image.open(image_path).convert("RGB")
        if item == 0 and DEBUG: print('DEBUG: Cityscapes {0:} -> image_path: {1:}'.format(self.split, image_path))
        ###

        ###
        gt_image_path = self.data_path + id_gt

        # gt_image = imageio.imread(gt_image_path,format='PNG-FI')[:,:,0]
        # gt_image = Image.fromarray(np.uint8(gt_image))

        gt_image = Image.open(gt_image_path)

        if item == 0 and DEBUG: print('DEBUG: Cityscapes {0:} -> gt_path: {1:}'.format(self.split, gt_image_path))
        ###

        if (self.split == "train" or self.split == "trainval") and self.training:
            image, gt_image = self._train_sync_transform(image, gt_image)
        else:
            image, gt_image = self._val_sync_transform(image, gt_image)

        return image, gt_image, item

    def _train_sync_transform(self, img, mask):
        if self.random_mirror: # default = True
            # random mirror
            if random.random() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                if mask: mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            crop_w, crop_h = self.crop_size

        if self.random_crop:  # default = False
            # random scale
            base_w , base_h = self.base_size
            w, h = img.size
            assert w >= h
            if (base_w / w) > (base_h / h):
                base_size = base_w 
                short_size = random.randint(int(base_size * 0.5), int(base_size * 2.0))
                ow = short_size
                oh = int(1.0 * h * ow / w)
            else:
                base_size = base_h
                short_size = random.randint(int(base_size * 0.5), int(base_size * 2.0))
                oh = short_size
                ow = int(1.0 * w * oh / h)

            img = img.resize((ow, oh), Image.BICUBIC)
            if mask: mask = mask.resize((ow, oh), Image.NEAREST)
            # pad crop
            if ow < crop_w or oh < crop_h:
                padh = crop_h - oh if oh < crop_h else 0
                padw = crop_w - ow if ow < crop_w else 0
                img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
                if mask: mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)
            # random crop crop_size
            w, h = img.size
            x1 = random.randint(0, w - crop_w)
            y1 = random.randint(0, h - crop_h)
            img = img.crop((x1, y1, x1 + crop_w, y1 + crop_h))
            if mask: mask = mask.crop((x1, y1, x1 + crop_w, y1 + crop_h))
        elif self.resize: # default = True
            img = img.resize(self.crop_size, Image.BICUBIC)
            if mask: mask = mask.resize(self.crop_size, Image.NEAREST)
        
        if self.gaussian_blur: # default = True
            # gaussian blur as in PSP
            if random.random() < 0.5:
                img = img.filter(ImageFilter.GaussianBlur(
                    radius=random.random()))
        # final transform
        if mask: 
            img, mask = self._img_transform(img), self._mask_transform(mask)
            return img, mask
        else:
            img = self._img_transform(img)
            return img

    def _val_sync_transform(self, img, mask):
        if self.random_crop:
            crop_w, crop_h = self.crop_size
            w, h = img.size
            if crop_w / w < crop_h / h:
                oh = crop_h
                ow = int(1.0 * w * oh / h)
            else:
                ow = crop_w
                oh = int(1.0 * h * ow / w)
            img = img.resize((ow, oh), Image.BICUBIC)
            mask = mask.resize((ow, oh), Image.NEAREST)
            # center crop
            w, h = img.size
            x1 = int(round((w - crop_w) / 2.))
            y1 = int(round((h - crop_h) / 2.))
            img = img.crop((x1, y1, x1 + crop_w, y1 + crop_h))
            mask = mask.crop((x1, y1, x1 + crop_w, y1 + crop_h))
        elif self.resize:
            img = img.resize(self.crop_size, Image.BICUBIC)
            mask = mask.resize(self.crop_size, Image.NEAREST)

        # final transform
        img, mask = self._img_transform(img), self._mask_transform(mask)
        return img, mask

    def _img_transform(self, image):
        if self.args.numpy_transform: # default = True
            image = np.asarray(image, np.float32)
            image = image[:, :, ::-1]  # change to BGR
            image -= IMG_MEAN
            image = image.transpose((2, 0, 1)).copy() # (C x H x W)
            new_image = torch.from_numpy(image)
        else:
            image_transforms = ttransforms.Compose([
                ttransforms.ToTensor(),
                ttransforms.Normalize([.485, .456, .406], [.229, .224, .225]),
            ])
            new_image = image_transforms(image)
        return new_image

    def _mask_transform(self, gt_image):
        target = np.asarray(gt_image, np.float32)
        target = self.id2trainId(target).copy()
        target = torch.from_numpy(target)

        return target

    def __len__(self):
        return len(self.items)

class City_DataLoader():
    def __init__(self, args, training=True):

        self.args = args

        data_set = City_Dataset(args,
                                data_root_path=self.args.data_root_path,
                                list_path=self.args.list_path,
                                split=args.split,
                                base_size=args.base_size,
                                crop_size=args.crop_size,
                                training=training,
                                class_16=args.class_16,
                                class_13=args.class_13)

        if (self.args.split == "train" or self.args.split == "trainval") and training:
            self.data_loader = data.DataLoader(data_set,
                                               batch_size=self.args.batch_size,
                                               shuffle=True,
                                               num_workers=self.args.data_loader_workers,
                                               pin_memory=self.args.pin_memory,
                                               drop_last=True)
        else:
            self.data_loader = data.DataLoader(data_set,
                                               batch_size=self.args.batch_size,
                                               shuffle=False,
                                               num_workers=self.args.data_loader_workers,
                                               pin_memory=self.args.pin_memory,
                                               drop_last=True)

        val_set = City_Dataset(args,
                               data_root_path=self.args.data_root_path,
                               list_path=self.args.list_path,
                               split='test' if self.args.split == "test" else 'val',
                               base_size=args.base_size,
                               crop_size=args.crop_size,
                               training=False,
                               class_16=args.class_16,
                               class_13=args.class_13)
        self.val_loader = data.DataLoader(val_set,
                                          batch_size=self.args.batch_size,
                                          shuffle=False,
                                          num_workers=self.args.data_loader_workers,
                                          pin_memory=self.args.pin_memory,
                                          drop_last=True)
        self.valid_iterations = (len(val_set) + self.args.batch_size) // self.args.batch_size

        self.num_iterations = (len(data_set) + self.args.batch_size) // self.args.batch_size

def flip(x, dim):
    dim = x.dim() + dim if dim < 0 else dim
    inds = tuple(slice(None, None) if i != dim
             else x.new(torch.arange(x.size(i)-1, -1, -1).tolist()).long()
             for i in range(x.dim()))
    return x[inds]

def inv_preprocess(imgs, num_images=1, img_mean=IMG_MEAN, numpy_transform=False):
    """Inverse preprocessing of the batch of images.
       
    Args:
      imgs: batch of input images.
      num_images: number of images to apply the inverse transformations on.
      img_mean: vector of mean colour values.
      numpy_transform: whether change RGB to BGR during img_transform.
  
    Returns:
      The batch of the size num_images with the same spatial dimensions as the input.
    """
    if numpy_transform:
        imgs = flip(imgs, 1)
    def norm_ip(img, min, max):
        img.clamp_(min=min, max=max)
        img.add_(-min).div_(max - min + 1e-5)
    norm_ip(imgs, float(imgs.min()), float(imgs.max()))
    return imgs

def decode_labels(mask, num_classes, num_images=1):
    """Decode batch of segmentation masks.
    
    Args:
      mask: result of inference after taking argmax.
      num_images: number of images to decode from the batch.
      num_classes: number of classes to predict.
    
    Returns:
      A batch with num_images RGB images of the same size as the input. 
    """

    assert num_classes == 16 or num_classes == 19
    label_colours = label_colours_16 if num_classes==16 else label_colours_19

    if isinstance(mask, torch.Tensor):
        mask = mask.data.cpu().numpy()
    n, h, w = mask.shape
    if n < num_images:
        num_images = n
    outputs = np.zeros((num_images, h, w, 3), dtype=np.uint8)
    for i in range(num_images):
      img = Image.new('RGB', (len(mask[i, 0]), len(mask[i])))
      pixels = img.load()
      for j_, j in enumerate(mask[i, :, :]):
          for k_, k in enumerate(j):
              if k < num_classes:
                  pixels[k_,j_] = label_colours[k]
      outputs[i] = np.array(img)
    return torch.from_numpy(outputs.transpose([0, 3, 1, 2]).astype('float32')).div_(255.0)
    
name_classes = [
    'road',
    'sidewalk',
    'building',
    'wall',
    'fence',
    'pole',
    'trafflight',
    'traffsign',
    'vegetation',
    'terrain',
    'sky',
    'person',
    'rider',
    'car',
    'truck',
    'bus',
    'train',
    'motorcycle',
    'bicycle',
    'unlabeled'
]
