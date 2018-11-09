from __future__ import print_function

from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torchvision.transforms as transforms
import torchvision.models as models
from model import run_style_transfer

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("content", type=str, help="Path of content image.")
parser.add_argument("style", type=str, help="Path of style image.")
parser.add_argument("output", type=str, help="Path of output image.")
parser.add_argument("--ws", type=float, default=1e6, help="Weight for style loss (default: 10^6).")
parser.add_argument("--wc", type=float, default=1, help="Weight for content loss (default: 1).")
parser.add_argument("--wsim", type=float, default=10, help="Weight for similarity loss (default: 10).")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# desired size of the output image
imsize = 512 if torch.cuda.is_available() else 128  # use small size if no gpu

loader = transforms.Compose([
    transforms.Resize((512,650)),  # scale imported image
    transforms.ToTensor()])  # transform it into a torch tensor


def image_loader(image_name):
    image = Image.open(image_name)
    # fake batch dimension required to fit network's input dimensions
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)


if __name__ == "__main__":

    style_img = image_loader(args.style)
    content_img = image_loader(args.content)

    print(style_img.size())
    print(content_img.size())

    assert style_img.size() == content_img.size(),     "we need to import style and content images of the same size"

    input_img = content_img.clone()
    # if you want to use white noise instead uncomment the below line:
    # input_img = torch.randn(content_img.data.size(), device=device)

    cnn = models.vgg19(pretrained=True).features.to(device).eval()
    cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

    output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                                content_img, style_img, input_img, device,
                                style_weight=args.ws, content_weight=args.wc, sim_weight=args.wsim)


    unloader = transforms.ToPILImage()  # reconvert into PIL image

    def imsave(tensor, path, title=None):
        image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
        image = image.squeeze(0)      # remove the fake batch dimension
        image = unloader(image)
        if title is not None:
            plt.title(title)
        plt.imsave(path, image)

    imsave(output, args.output)
