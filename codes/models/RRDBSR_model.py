

import sys
sys.path.append('../')
import logging
from collections import OrderedDict
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
import models.networks as networks
import models.lr_scheduler as lr_scheduler
from .base_model import BaseModel
from .loss import SRLoss
import utils.util as util
import itertools
import torch.nn.functional as F

logger = logging.getLogger('base')


class RRDBSRModel(BaseModel):
    def __init__(self, opt):
        super(RRDBSRModel, self).__init__(opt)



        # define networks and load pretrained models
        train_opt = opt['train']

        self.netG_SR = networks.define_SR(opt).to(self.device)

        if self.is_train:
            if not self.opt['full_sr']:

                self.netG_BA = networks.define_G(opt).to(self.device)



        if opt['dist']:
            self.netG_SR = DistributedDataParallel(self.netG_SR, device_ids=[torch.cuda.current_device()],find_unused_parameters=True)
            if self.is_train:
                if not self.opt['full_sr']:
                    self.netG_BA = DistributedDataParallel(self.netG_BA, device_ids=[torch.cuda.current_device()],find_unused_parameters=True)


        else:

            self.netG_SR = DataParallel(self.netG_SR)
            if self.is_train:
                if not self.opt['full_sr']:
                    self.netG_BA = DataParallel(self.netG_BA)



        # define losses, optimizer and scheduler
        if self.is_train:


            # losses
            self.criterion = SRLoss(train_opt['loss_type']).to(self.device)  # define GAN loss.

            # optimizers
            self.optimizer_G = torch.optim.Adam(self.netG_SR.parameters(),lr=train_opt['lr'], betas=(train_opt['beta1'], train_opt['beta2']))

            self.optimizers.append(self.optimizer_G)


            #scheduler

            if train_opt['lr_scheme'] == 'MultiStepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.MultiStepLR_Restart(optimizer, train_opt['lr_steps'],
                                                         restarts=train_opt['restarts'],
                                                         weights=train_opt['restart_weights'],
                                                         gamma=train_opt['lr_gamma'],
                                                         clear_state=train_opt['clear_state']))
            else:
                raise NotImplementedError("lr_scheme does not implement still")


            self.log_dict = OrderedDict()


        self.load()  # load pre-trained mode

        if self.is_train:
            self.train_state()

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
    def feed_data(self, data):
        self.LR = data['LQ'].to(self.device)
        self.HR = data['HQ'].to(self.device)

    def B2A(self):

        with torch.no_grad():
            fake_LR=self.netG_BA(self.LR)

        return fake_LR

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        if self.opt['full_sr']:
            self.fake_LR = self.LR
        else:
            self.fake_LR = self.B2A()
        self.SR = self.netG_SR(self.fake_LR)


    def backward_G(self,step):
        """Calculate the loss for generators G_A and G_B"""



        self.loss_G=self.criterion(self.SR,self.HR)
        if len(self.loss_G)!=1:
            if self.opt['other_step']>step:
                self.loss_total=self.loss_G[0]+\
                                self.loss_G[1]*self.opt['l_other_weight']
            else:
                self.loss_total=self.loss_G[0]
        else:
            self.loss_total=self.loss_G[0]
        self.loss_total.backward()


    def optimize_parameters(self, step):
        # G
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()  # compute fake images and reconstruction images.
        self.optimizer_G.zero_grad()
        self.backward_G(step)  # calculate gradients for G_A and G_B
        self.optimizer_G.step()  # update G_A and G_B's weights

        # set log
        for i in range(len(self.loss_G)):
            self.log_dict[str(i)] = self.loss_G[i].item()

    def train_state(self):
        self.netG_SR.train()
        if not self.opt['full_sr']:
            self.netG_BA.eval()


    def test_state(self):
        self.netG_SR.eval()
        if not self.opt['full_sr']:
            self.netG_BA.eval()


    def val(self):
        self.test_state()
        with torch.no_grad():
            self.forward()
        self.train_state()

    def test(self):
        self.netG_SR.eval()
        with torch.no_grad():
            SR=self.netG_SR(self.LR)
        return {'SR':SR}



    def get_current_log(self):
        return self.log_dict



    def print_network(self):


        if self.is_train:
            # Generator
            s, n = self.get_network_description(self.netG_SR)
            net_struc_str = '{} - {}'.format(self.netG_SR.__class__.__name__,
                                             self.netG_SR.module.__class__.__name__)
            logger.info('Network G structure: {}, with parameters: {:,d}'.format(
                net_struc_str, n))
            logger.info(s)





    def load(self):
        load_path_G_SR = self.opt['path']['pretrain_model_G_SR']
        load_path_G_BA = self.opt['path']['pretrain_model_G_BA']

        if load_path_G_BA is not None:
            logger.info('Loading models for G [{:s}] ...'.format(load_path_G_BA))
            self.load_network(load_path_G_BA, self.netG_BA, self.opt['path']['strict_load'])
        else:
            logger.info('GAN model does not exist!')
            if self.is_train:
                if not self.opt['full_sr']:
                    exit(1)
        if load_path_G_SR is not None:

            logger.info('Loading models for D [{:s}] ...'.format(load_path_G_SR))
            self.load_network(load_path_G_SR, self.netG_SR, self.opt['path']['strict_load'])

    def save(self, iter_step):
        self.save_network(self.netG_SR, 'G_SR', iter_step)
