import random
import numpy as np

import lmdb
import torch
import torch.utils.data as data
import data.util as util


class SPDataset(data.Dataset):
    '''Read LQ images only in the test phase.'''

    def __init__(self, opt):
        super(SPDataset, self).__init__()
        self.opt = opt
        self.data_type = self.opt['data_type']
        self.crop_size = self.opt['crop_size']
        self.LQ_env = None  # environment for lmdb
        self.HQ_env = None  # environment for lmdb
        self.paths_LQ, self.sizes_LQ = util.get_image_paths(self.data_type, opt['dataroot_LQ'])
        self.paths_HQ, self.sizes_HQ = util.get_image_paths(self.data_type, opt['dataroot_HQ'])
        assert self.paths_LQ, 'Error: LQ paths are empty.'
        assert self.paths_HQ, 'Error: HQ paths are empty.'

    def _init_lmdb(self):
        self.HQ_env = lmdb.open(self.opt['dataroot_GT'], readonly=True, lock=False, readahead=False,
                                meminit=False)
        self.LQ_env = lmdb.open(self.opt['dataroot_LQ'], readonly=True, lock=False, readahead=False,
                                meminit=False)


    def __getitem__(self, index):
        if self.data_type == 'lmdb' and self.LQ_env is None and self.HQ_env is None:
            self._init_lmdb()


        # get LQ and HQ image
        LQ_path = self.paths_LQ[index]
        HQ_path = self.paths_HQ[index]
        LQ_resolution = [int(s) for s in self.sizes_LQ[index].split('_')
                      ] if self.data_type == 'lmdb' else None
        HQ_resolution = [int(s) for s in self.sizes_HQ[index].split('_')
                      ] if self.data_type == 'lmdb' else None
        img_LQ = util.read_img(self.LQ_env, LQ_path, LQ_resolution)
        img_HQ = util.read_img(self.HQ_env, HQ_path, HQ_resolution)


        if self.opt['crop_size']:
            # randomly crop
            LH,LW,_=img_LQ.shape
            rnd_h = random.randint(0, max(0, LH - self.crop_size))
            rnd_w = random.randint(0, max(0, LW - self.crop_size))
            if self.opt['SR']:
                rnd_hh=rnd_h * self.opt['scale']
                rnd_wh = rnd_w * self.opt['scale']
                patch_size=self.crop_size*self.opt['scale']
                img_LQ = img_LQ[rnd_h:rnd_h + self.crop_size, rnd_w:rnd_w + self.crop_size, :]
                img_HQ = img_HQ[rnd_hh:rnd_hh + patch_size, rnd_wh:rnd_wh + patch_size, :]
            else:
                img_LQ = img_LQ[rnd_h:rnd_h + self.crop_size, rnd_w:rnd_w + self.crop_size, :]
                img_HQ = img_HQ[rnd_h:rnd_h + self.crop_size, rnd_w:rnd_w + self.crop_size, :]

        if self.opt['phase']=='train':
            # augmentation - flip, rotate
            img_LQ, img_HQ = util.augment([img_LQ, img_HQ], self.opt['use_flip'],
                                          self.opt['use_rot'])

        # BGR to RGB, HWC to CHW, numpy to tensor
        if img_LQ.shape[2] == 3:
            img_LQ = img_LQ[:, :, [2, 1, 0]]
        if img_HQ.shape[2] == 3:
            img_HQ = img_HQ[:, :, [2, 1, 0]]

        img_LQ = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LQ, (2, 0, 1)))).float()
        img_HQ = torch.from_numpy(np.ascontiguousarray(np.transpose(img_HQ, (2, 0, 1)))).float()

        return {'LQ': img_LQ, 'LQ_path': LQ_path,'HQ':img_HQ,'HQ_path':HQ_path}

    def __len__(self):
        return len(self.paths_LQ)
