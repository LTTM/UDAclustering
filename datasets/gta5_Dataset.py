# -*- coding: utf-8 -*-
from PIL import Image, ImageFile
import numpy as np
import os
import torch.utils.data as data
import imageio

from datasets.cityscapes_Dataset import City_Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True
DEBUG = False

class GTA5_Dataset(City_Dataset):
    def __init__(self,
                 args,
                 data_root_path='./datasets/GTA5',
                 list_path='./datasets/GTA5',
                 split='train',
                 base_size=769,
                 crop_size=769,
                 training=True):

        self.args = args
        self.data_path=data_root_path
        self.list_path=list_path
        self.split=split
        if DEBUG: print('DEBUG: GTA {0:} dataset path is {1:}'.format(self.split, self.list_path))
        self.base_size=base_size
        self.crop_size=crop_size
        if DEBUG: print('DEBUG: GTA {0:} dataset image size is {1:}'.format(self.split, self.crop_size))

        self.base_size = self.base_size if isinstance(self.base_size, tuple) else (self.base_size, self.base_size)
        self.crop_size = self.crop_size if isinstance(self.crop_size, tuple) else (self.crop_size, self.crop_size)
        self.training = training

        self.random_mirror = args.random_mirror
        self.random_crop = args.random_crop
        self.resize = args.resize
        self.gaussian_blur = args.gaussian_blur

        ###
        item_list_filepath = os.path.join(self.list_path, self.split + ".txt")
        self.items = [id for id in open(item_list_filepath)]
        ###

        ignore_label = -1

        ###
        # self.id_to_trainid = {i: i for i in range(-1, 19)}
        # self.id_to_trainid[255] = ignore_label

        self.id_to_trainid = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
                              19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
                              26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}
        ###

        self.class_16 = False
        self.class_13 = False

        if DEBUG: print('DEBUG: GTA {0:} -> item_list_filepath: {1:} , first item: {2:}'.format(self.split, item_list_filepath, self.items[0]))
        if DEBUG: print("{} num images in GTA5 {} set have been loaded.".format(len(self.items), self.split))

    def __getitem__(self, item):
        ###
        id_img, id_gt = self.items[item].strip('\n').split(' ')
        image_path = self.data_path + id_img
        image = Image.open(image_path).convert("RGB")
        if item == 0 and DEBUG: print('DEBUG: GTA {0:} -> image_path: {1:}'.format(self.split, image_path))
        ###

        ###
        gt_image_path = self.data_path + id_gt

        # gt_image = imageio.imread(gt_image_path, format='PNG-FI')[:, :, 0]
        # gt_image = Image.fromarray(np.uint8(gt_image))
        #
        gt_image = Image.open(gt_image_path)

        if item == 0 and DEBUG: print('DEBUG: GTA {0:} -> image_path: {1:}'.format(self.split, image_path))
        ###

        if (self.split == "train" or self.split == "trainval" or self.split =="all") and self.training:
            image, gt_image = self._train_sync_transform(image, gt_image)
        else:
            image, gt_image = self._val_sync_transform(image, gt_image)

        return image, gt_image, item

class GTA5_DataLoader():
    def __init__(self, args, training=True):

        self.args = args

        data_set = GTA5_Dataset(args, 
                                data_root_path=args.data_root_path,
                                list_path=args.list_path,
                                split=args.split,
                                base_size=args.base_size,
                                crop_size=args.crop_size,
                                training=training)

        if self.args.split == "train" or self.args.split == "trainval" or self.args.split =="all":
            self.data_loader = data.DataLoader(data_set,
                                               batch_size=self.args.batch_size,
                                               shuffle=True,
                                               num_workers=self.args.data_loader_workers,
                                               pin_memory=self.args.pin_memory,
                                               drop_last=True)
        elif self.args.split =="val" or self.args.split == "test":
            self.data_loader = data.DataLoader(data_set,
                                               batch_size=self.args.batch_size,
                                               shuffle=False,
                                               num_workers=self.args.data_loader_workers,
                                               pin_memory=self.args.pin_memory,
                                               drop_last=True)
        else:
            raise Warning("split must be train/val/trainavl/test/all")

        val_split = 'val' if self.args.split == "train" else 'test'
        val_set = GTA5_Dataset(args, 
                            data_root_path=args.data_root_path,
                            list_path=args.list_path,
                            split=val_split,
                            base_size=args.base_size,
                            crop_size=args.crop_size,
                            training=False)
        self.val_loader = data.DataLoader(val_set,
                                            batch_size=self.args.batch_size,
                                            shuffle=False,
                                            num_workers=self.args.data_loader_workers,
                                            pin_memory=self.args.pin_memory,
                                            drop_last=True)
        self.valid_iterations = (len(val_set) + self.args.batch_size) // self.args.batch_size

        self.num_iterations = (len(data_set) + self.args.batch_size) // self.args.batch_size


        