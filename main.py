from __future__ import print_function
import argparse
import matlab
from matlab.engine import start_matlab
from PIL import Image
import numpy as np
import os

import torch
import torchvision.transforms as transforms
import torchvision.models as models
from model import run_style_transfer
from segment import add_arguments


parser = argparse.ArgumentParser()
parser.add_argument("content", type=str, help="Path of content image.")
parser.add_argument("style", type=str, help="Path of style image.")
parser.add_argument("output", type=str, help="Path of output image.")
parser.add_argument("--masks", type=str, help="Path of masks to load.")
parser.add_argument("--lr", type=float, default=1.0, help="Initial learning rate.")
parser.add_argument("--iters", type=int, default=300, help="Number of iterations to run.")
parser.add_argument("--post_s", type=float, default=60.0, help="sigma_s for post processing recursive filter. (default: 60)")
parser.add_argument("--post_r", type=float, default=1.0, help="sigma_r for post processing recursive filter. (default: 1)")
parser.add_argument("--post_it", type=int, default=3, help="Number of iterations for post processing recursive filter. (default: 3)")
parser.add_argument("--ws", type=float, default=1e6, help="Weight for style loss (default: 10^6).")
parser.add_argument("--wc", type=float, default=1, help="Weight for content loss (default: 1).")
parser.add_argument("--wsim", type=float, default=10, help="Weight for similarity loss (default: 10).")
add_arguments(parser)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# desired size of the output image
imsize = 512 if torch.cuda.is_available() else 128  # use small size if no gpu

loader = transforms.Compose([
    transforms.Resize((468,700)),  # scale imported image
    transforms.ToTensor()])  # transform it into a torch tensor


def image_loader(image_name):
    image = Image.open(image_name)
    # fake batch dimension required to fit network's input dimensions
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)


def load_masks():
    if args.masks is None:
        # create masks
        from segment import segment
        masks = segment(args)
    else:
        # load masks
        masks = torch.load(args.masks)

    style_mask = masks["tar"]
    content_mask = masks["in"]
    style_mask = style_mask.to(device).unsqueeze(1)
    content_mask = content_mask.to(device).unsqueeze(1)
    return style_mask, content_mask


if __name__ == "__main__":

    style_img = image_loader(args.style)
    content_img = image_loader(args.content)

    print(style_img.size())
    print(content_img.size())

    assert style_img.size() == content_img.size(),     "we need to import style and content images of the same size"

    input_img = content_img.clone()
    # if you want to use white noise instead uncomment the below line:
    # input_img = torch.randn(content_img.data.size(), device=device)

    style_mask, content_mask = load_masks()

    cnn = models.vgg19(pretrained=True).features.to(device).eval()
    cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

    output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                                content_img, style_img, input_img, style_mask, content_mask, device,
                                lr=args.lr, num_steps=args.iters,
                                style_weight=args.ws, content_weight=args.wc, sim_weight=args.wsim)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    def unload(tensor):
        # Convert an image from tensor to PIL image
        image = tensor.cpu().clone()
        image = image.squeeze(0)
        image_pil = transforms.functional.to_pil_image(image)
        return image_pil

    output = unload(output)
    save_path = args.output
    plt.imsave(save_path, output)
    print("Save image to {}".format(save_path))


    print("Post processing")
    inimg = np.array(unload(content_img))
    inimg_mat = matlab.int32(inimg.tolist())
    outimg = np.array(output)
    outimg_mat = matlab.int32(outimg.tolist())

    eng = start_matlab()
    processed_img = inimg - np.asarray(eng.RF(inimg_mat, args.post_s, args.post_r, args.post_it, inimg_mat)) + \
            np.asarray(eng.RF(outimg_mat, args.post_s, args.post_r, args.post_it, inimg_mat))
    processed_img = np.uint8(np.clip(processed_img, 0, 255))

    output_rt, output_ext = os.path.splitext(args.output)
    save_path = output_rt + "_post" + output_ext
    print("Save post processed image to {}".format(save_path))
    plt.imsave(save_path, Image.fromarray(processed_img))
    
