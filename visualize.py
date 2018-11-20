import torch
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

masks = torch.load("masks.pth")
mask_tensors = [masks["tar"], masks["in"]]

for mask, save_root in zip(mask_tensors, ["style/", "content/"]):
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    
    for tensor, idx in zip(mask, masks["categories"]):
        path = os.path.join(save_root, "{}.png".format(idx + 1))    # convert 0-index to 1-index
        plt.imsave(path, tensor.numpy())