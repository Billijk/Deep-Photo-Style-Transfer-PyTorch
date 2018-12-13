from __future__ import print_function
import argparse
import matlab
from matlab.engine import start_matlab
from PIL import Image
import numpy as np
import os

import torch
import torchvision.transforms.functional as t_func
import torchvision.models as models
import torch.optim as optim
from model import get_style_model_and_losses, matting_regularizer
from segment import add_arguments

parser = argparse.ArgumentParser()
parser.add_argument("content", type=str, help="Path of content image.")
parser.add_argument("style", type=str, help="Path of style image.")
parser.add_argument("output", type=str, help="Path of output image.")
parser.add_argument("--step", type=int, default=300, help="Number of steps to optimize.")
parser.add_argument("--size", type=int, default=480, help="Size for scaling image.")
parser.add_argument("--masks", type=str, help="Path of masks to load.")
parser.add_argument("--lr", type=float, default=1.0, help="Initial learning rate.")
parser.add_argument("--iters", type=int, default=300, help="Number of iterations to run.")
parser.add_argument("--laplacian", type=str, default="", help="Load pre-calculated matting laplacian values."
        "If this is not specified, then a new values will be calculated and saved on disk.")
parser.add_argument("--wr", type=float, default=1e4, help="Weight for photorealism regularization (default: 10000).")
parser.add_argument("--ws", type=float, default=1e4, help="Weight for style loss (default: 10000).")
parser.add_argument("--wc", type=float, default=1, help="Weight for content loss (default: 1).")
parser.add_argument("--wsim", type=float, default=10, help="Weight for similarity loss (default: 10).")
add_arguments(parser)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def image_loader(image_name, h, w=None):
    image = Image.open(image_name)
    if w is not None: size = (h, w)
    else: size = h
    size = (468, 700)
    image = t_func.resize(image, size)
    # fake batch dimension required to fit network's input dimensions
    image = t_func.to_tensor(image).unsqueeze(0)
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

    style_img = image_loader(args.style, args.size)
    content_img = image_loader(args.content, style_img.size(2), style_img.size(3))
    print(style_img.size())
    print(content_img.size())

    if len(args.laplacian) > 0:
        print("Loading laplacian from {}".format(args.laplacian))
        laplacian_i, laplacian_v = torch.load(args.laplacian)
        s = style_img.size(2) * style_img.size(3)
        laplacian_size = torch.Size([s, s])
        laplacian_mat = torch.sparse_coo_tensor(laplacian_i, laplacian_v, size=laplacian_size).to(device)
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
        laplacian_i = torch.stack([laplacian_ix, laplacian_iy], dim=0)
        laplacian_mat = torch.sparse_coo_tensor(laplacian_i, laplacian_v, size=laplacian_size).to(device)

        save_path = args.output + ".ml_{}.pth".format(s)
        torch.save((laplacian_i, laplacian_v), save_path)
        print("Values for matting laplacian saved at {}. Load this for the next time so you don't have to compute it again.".format(save_path))


    assert style_img.size() == content_img.size(),     "we need to import style and content images of the same size"
    assert laplacian_mat.size(0) == style_img.size(2) * style_img.size(3)

    input_img = content_img.clone()
    # if you want to use white noise instead uncomment the below line:
    # input_img = torch.randn(content_img.data.size(), device=device)

    style_mask, content_mask = load_masks()

    cnn = models.vgg19(pretrained=True).features.to(device).eval()
    cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

    print('Building the style transfer model..')
    model, style_losses, content_losses, sim_losses = get_style_model_and_losses(cnn,
        cnn_normalization_mean, cnn_normalization_std, style_img, content_img,
        style_mask, content_mask, device)
    optimizer = optim.LBFGS([input_img.requires_grad_()], lr=args.lr)
    regularizer = matting_regularizer.apply

    print('Optimizing..')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    def imsave(tensor, path, title=None):
        image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
        image.data.clamp_(0, 1)
        image = image.squeeze(0)      # remove the fake batch dimension
        image = t_func.to_pil_image(image)
        if title is not None:
            plt.title(title)
        plt.imsave(path, image)

    run = [0]
    while run[0] <= args.step:

        def closure():

            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0
            sim_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss
            for siml in sim_losses:
                sim_score += siml.loss

            style_score *= args.ws
            content_score *= args.wc
            sim_score *= args.wsim
            regularize_score = args.wr * regularizer(laplacian_mat, input_img)

            loss = style_score + content_score + sim_score + regularize_score
            loss.backward()

            run[0] += 1
            if run[0] % 100 == 0:
                print("run {}:".format(run))
                print('Style Loss : {:4f} Content Loss: {:4f} Similarity Loss: {:4f} Regularization: {:4f}'.format(
                    style_score.item(), content_score.item(), sim_score.item(), regularize_score))
                suffix_pos = args.output.rindex('.')
                image_save_path = args.output[:suffix_pos] + "_it{}".format(run[0]) + args.output[suffix_pos:]
                imsave(input_img, image_save_path)

            return loss

        optimizer.step(closure)
        # correct the values of updated input image
        input_img.data.clamp_(0, 1)

    print("Finished.")
