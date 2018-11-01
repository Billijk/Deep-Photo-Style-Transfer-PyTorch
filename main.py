from __future__ import print_function
import argparse
import matlab
from matlab.engine import start_matlab
from PIL import Image
import numpy as np

import torch
import torchvision.transforms as transforms
import torchvision.models as models
from model import run_style_transfer

parser = argparse.ArgumentParser()
parser.add_argument("content", type=str, help="Path of content image.")
parser.add_argument("style", type=str, help="Path of style image.")
parser.add_argument("output", type=str, help="Path of output image.")
parser.add_argument("--size", type=int, default=512, help="Size for scaling image.")
parser.add_argument("--laplacian", type=str, default="", help="Load pre-calculated matting laplacian values."
        "If this is not specified, then a new values will be calculated and saved on disk.")
parser.add_argument("--wr", type=float, default=1e4, help="Weight for photorealism regularization (default: 10000).")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

loader = transforms.Compose([
    transforms.Resize(args.size),  # scale imported image
    transforms.ToTensor()])  # transform it into a torch tensor


def image_loader(image_name):
    image = Image.open(image_name)
    # fake batch dimension required to fit network's input dimensions
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)


if __name__ == "__main__":

    style_img = image_loader(args.style)
    content_img = image_loader(args.content)

    if len(args.laplacian) > 0:
        print("Loading laplacian from {}".format(args.laplacian))
        laplacian_mat = torch.load(args.laplacian).to(device)
    else:
        print("Calculating laplacian")
        # call matlab to compute matting laplacian
        eng = start_matlab()
        eng.cd("gen_laplacian")
        laplacian = np.asarray(eng.gen_laplacian("../" + args.content, style_img.size(2), style_img.size(3)))
        laplacian = torch.Tensor(laplacian)
        laplacian_ix = laplacian[:, 0].long() - 1
        laplacian_iy = laplacian[:, 1].long() - 1
        laplacian_v = laplacian[:, 2]
        s = style_img.size(2) * style_img.size(3)
        laplacian_size = torch.Size([s, s])
        laplacian_mat = torch.sparse_coo_tensor(torch.stack([laplacian_ix, laplacian_iy], dim=0), laplacian_v, size=laplacian_size)
        laplacian_mat = laplacian_mat.to(device)

        save_path = args.content + ".ml_{}.pth".format(s)
        torch.save(laplacian_mat, save_path)
        print("Values for matting laplacian saved at {}. Load this for the next time so you don't have to compute it again.".format(save_path))

    print(style_img.size())
    print(content_img.size())
    print(laplacian_mat.size())

    assert style_img.size() == content_img.size(),     "we need to import style and content images of the same size"
    assert laplacian_mat.size(0) == style_img.size(2) * style_img.size(3)

    input_img = content_img.clone()
    # if you want to use white noise instead uncomment the below line:
    # input_img = torch.randn(content_img.data.size(), device=device)

    cnn = models.vgg19(pretrained=True).features.to(device).eval()
    cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

    output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                                content_img, style_img, input_img, laplacian_mat,
                                args.wr, device)


    unloader = transforms.ToPILImage()  # reconvert into PIL image

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    def imsave(tensor, path, title=None):
        image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
        image = image.squeeze(0)      # remove the fake batch dimension
        image = unloader(image)
        if title is not None:
            plt.title(title)
        plt.imsave(path, image)

    imsave(output, args.output)
