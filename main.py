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
                                content_img, style_img, input_img, device)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    def unload(tensor):
        # Convert an image from tensor to numpy array in shape (H, W, C)
        image = output.cpu().clone()
        image = image.squeeze(0)
        image_pil = transforms.functional.to_pil_image(image)
        image = np.array(image_pil)
        return image

    print("Post processing")
    inimg = unload(content_img)
    inimg_mat = matlab.uint8(inimg.tolist())
    outimg = unload(output)
    outimg_mat = matlab.uint8(outimg.tolist())
    #import pdb; pdb.set_trace()

    eng = start_matlab()
    processed_img = outimg - np.asarray(eng.RF(inimg_mat, 60., 1., 3, inimg_mat)) + \
            np.asarray(eng.RF(outimg_mat, 60., 1., 3, inimg_mat))
    plt.imsave(args.output, Image.fromarray(np.uint8(processed_img)))
    
