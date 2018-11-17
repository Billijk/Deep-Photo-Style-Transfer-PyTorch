from __future__ import print_function

from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from model import run_style_transfer

# System libs
import os
import datetime
import argparse
from distutils.version import LooseVersion
# Numerical libs
import numpy as np
import torch
import torch.nn as nn
# Our libs
from dataset import TestDataset
from models import ModelBuilder, SegmentationModule
from lib.nn import async_copy_to

def segment_img(net, data, args, valid_masks=None, cutoff=0.2):
    """
    return Tensor (Categories, H, W)
    """
    segSize = (468, 700) # TODO: change this using input arguments
    img_resized_list = data['img_data']
    pred = torch.zeros(1, args.num_class, segSize[0], segSize[1])
    for img in img_resized_list:
        feed_dict = data.copy()
        feed_dict['img_data'] = img
        del feed_dict['img_ori']
        del feed_dict['info']
        feed_dict = async_copy_to(feed_dict, 0)

        # forward pass
        pred_tmp = net(feed_dict, segSize=segSize)
        pred = pred + pred_tmp.cpu() / len(args.imgSize)

    if valid_masks is not None:
        mask = torch.zeros(1, args.num_class, segSize[0], segSize[1])
        mask[:, valid_masks, :, :] = 1
        pred *= mask
        pred = pred / (pred.max(dim=0)[0] - pred.min(dim=0)[0])

    # cut off
    pred[pred < cutoff] = 0
    return pred.squeeze()

def test(segmentation_module, data, args):
    tar_seg = segment_img(segmentation_module, data["tar"], args)
    valid_categories = np.unique(tar_seg.numpy().nonzero()[0])
    # input image can only be segmented with categories in target image
    in_seg = segment_img(segmentation_module, data["in"], args, valid_categories)
    # only keep valid category layers
    in_seg = in_seg[valid_categories]
    tar_seg = tar_seg[valid_categories]
    print("Categories: ", valid_categories)
    return {"in": in_seg, "tar": tar_seg, "categories": valid_categories}

def load_data(data_dict):
    data_list = [{'fpath_img': data_dict["in"]}, {'fpath_img': data_dict["tar"]}]
    dset = TestDataset(data_list, args)
    return {"in": dset[0], "tar": dset[1]}

def segment(args):
    builder = ModelBuilder()
    net_encoder = builder.build_encoder(
        arch=args.arch_encoder,
        fc_dim=args.fc_dim,
        weights=args.weights_encoder)
    net_decoder = builder.build_decoder(
        arch=args.arch_decoder,
        fc_dim=args.fc_dim,
        num_class=args.num_class,
        weights=args.weights_decoder,
        use_softmax=True)

    crit = nn.NLLLoss(ignore_index=-1)

    segmentation_module = SegmentationModule(net_encoder, net_decoder, crit)
    segmentation_module.eval()
    segmentation_module.cuda()

    # Dataset and Loader
    data_dict = {"in": args.in_img, "tar": args.tar_img}
    data = load_data(data_dict)

    # Main loop
    with torch.no_grad():
        res = test(segmentation_module, data, args)
    return res
    

def main(args):
    
    res = segment(args)
    torch.save(res, args.save_path)
    print('Inference done!')


if __name__ == '__main__':
    assert LooseVersion(torch.__version__) >= LooseVersion('0.4.0'), \
        'PyTorch>=0.4.0 is required'

    parser = argparse.ArgumentParser()
    # Path related arguments
    parser.add_argument('--in_img', required=True)
    parser.add_argument('--tar_img', required=True)
    parser.add_argument('--save_path', required=True)
    parser.add_argument('--model_path', required=True,
                        help='folder to model path')
    parser.add_argument('--suffix', default='_epoch_20.pth',
                        help="which snapshot to load")

    # Model related arguments
    parser.add_argument('--arch_encoder', default='resnet50_dilated8',
                        help="architecture of net_encoder")
    parser.add_argument('--arch_decoder', default='ppm_bilinear_deepsup',
                        help="architecture of net_decoder")
    parser.add_argument('--fc_dim', default=2048, type=int,
                        help='number of features between encoder and decoder')

    # Data related arguments
    parser.add_argument('--num_val', default=-1, type=int,
                        help='number of images to evalutate')
    parser.add_argument('--num_class', default=150, type=int,
                        help='number of classes')
    parser.add_argument('--batch_size', default=1, type=int,
                        help='batchsize. current only supports 1')
    parser.add_argument('--imgSize', default=[300, 400, 500, 600],
                        nargs='+', type=int,
                        help='list of input image sizes.'
                             'for multiscale testing, e.g. 300 400 500')
    parser.add_argument('--imgMaxSize', default=1000, type=int,
                        help='maximum input image size of long edge')
    parser.add_argument('--padding_constant', default=8, type=int,
                        help='maxmimum downsampling rate of the network')
    parser.add_argument('--segm_downsampling_rate', default=8, type=int,
                        help='downsampling rate of the segmentation label')
    """
    parser.add_argument("content", type=str, help="Path of content image.")
    parser.add_argument("style", type=str, help="Path of style image.")
    parser.add_argument("output", type=str, help="Path of output image.")
    """
    args = parser.parse_args()
    print(args)

    

    # absolute paths of model weights
    args.weights_encoder = os.path.join(args.model_path,
                                        'encoder' + args.suffix)
    args.weights_decoder = os.path.join(args.model_path,
                                        'decoder' + args.suffix)

    assert os.path.exists(args.weights_encoder) and \
        os.path.exists(args.weights_encoder), 'checkpoint does not exitst!'

    main(args)
