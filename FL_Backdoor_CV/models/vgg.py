'''
    dummy file to use as an adaptor to switch between
    two vgg architectures

    vgg9: use vgg9_only.py which is from https://github.com/kuangliu/pytorch-cifar
    vgg11/13/16/19: use vgg_modified.py which is modified from https://github.com/pytorch/vision.git
'''

import torch
import torch.nn as nn
import models.vgg9_only as vgg9
import models.vgg_modified as vgg_mod
import logging

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def get_vgg_model(vgg_name):
    logging.info("GET_VGG_MODEL: Fetch {}".format(vgg_name))
    if vgg_name == 'vgg9':
        return vgg9.VGG('VGG9')
    elif vgg_name == 'vgg11':
        return vgg_mod.vgg11()
    elif vgg_name == 'vgg13':
        return vgg_mod.vgg13()
    elif vgg_name == 'vgg16':
        return vgg_mod.vgg16()

