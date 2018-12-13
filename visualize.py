import torch
import os
import sys
from scipy.io import loadmat
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

masks = torch.load(sys.argv[1])

def colorEncode(labelmap, colors, mode='BGR'):
    labelmap = labelmap.astype('int')
    labelmap_rgb = np.zeros((labelmap.shape[0], labelmap.shape[1], 3),
                            dtype=np.uint8)
    for label in unique(labelmap):
        if label < 0:
            continue
        labelmap_rgb += (labelmap == label)[:, :, np.newaxis] * \
            np.tile(colors[label],
                    (labelmap.shape[0], labelmap.shape[1], 1))

    if mode == 'BGR':
        return labelmap_rgb[:, :, ::-1]
    else:
        return labelmap_rgb

def visualize_result(preds, save_name):
    colors = loadmat('data/color150.mat')['colors']
    pred_color = colorEncode(preds, colors)
    plt.imsave("{}.png".format(save_name), pred_color)

visualize_result(masks["tar"], "style")
visualize_result(masks["in"], "content")