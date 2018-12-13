import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import copy

class ContentLoss(nn.Module):

    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        # we 'detach' the target content from the tree used
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input


class SimilarityLoss(nn.Module):
    def __init__(self, target):
        super().__init__()
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.l1_loss(input, self.target)
        return input


def gram_matrix(input):
    a, b, c, d = input.size()  # a=mask layer size
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)

class StyleLoss(nn.Module):

    def __init__(self, target_feature, style_mask, content_mask):
        """
        Target feature: 1 x channel x H x W
        Mask: Layer x 1 x H x W
        """
        super(StyleLoss, self).__init__()
        self.style_mask = style_mask
        self.content_mask = content_mask
        self.target = target_feature.detach()
        self.C = set()
        for elem in self.style_mask.view(-1):
            self.C.add(elem.item())
        print(self.C)

    def forward(self, input):
        self.loss = 0
        for c in self.C:
            M_c_I = (self.content_mask == c).to(torch.float)
            M_c_S = (self.style_mask == c).to(torch.float)
            G_c_O = gram_matrix(input * M_c_I)
            G_c_S = gram_matrix(self.target * M_c_S)
            self.loss += F.mse_loss(G_c_O, G_c_S)   
        return input

# create a module to normalize input image so we can easily put it in a
# nn.Sequential
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std


# desired depth layers to compute style/content losses :
sim_layers_default = ['conv_1', 'conv_2', 'conv_3']
content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img, content_img, style_mask, content_mask, device,
                               sim_layers=sim_layers_default,
                               content_layers=content_layers_default,
                               style_layers=style_layers_default):
    cnn = copy.deepcopy(cnn)

    # normalization module
    normalization = Normalization(normalization_mean, normalization_std).to(device)

    # just in order to have an iterable access to or list of content/syle
    # losses
    sim_losses = []
    content_losses = []
    style_losses = []

    # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
    # to put in modules that are supposed to be activated sequentially
    model = nn.Sequential(normalization)

    i = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            # The in-place version doesn't play very nicely with the ContentLoss
            # and StyleLoss we insert below. So we replace with out-of-place
            # ones here.
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
            content_mask = layer(content_mask)
            style_mask = layer(style_mask)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in sim_layers:
            # add similarity loss:
            target = model(content_img).detach()
            sim_loss = SimilarityLoss(target)
            model.add_module("sim_loss_{}".format(i), sim_loss)
            sim_losses.append(sim_loss)

        if name in content_layers:
            # add content loss:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            # add style loss:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature, style_mask, content_mask)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    # now we trim off the layers after the last content and style losses
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]

    return model, style_losses, content_losses, sim_losses


def get_input_optimizer(input_img, lr):
    # this line to show that input is a parameter that requires a gradient
    optimizer = optim.LBFGS([input_img.requires_grad_()], lr=lr)
    return optimizer

def run_style_transfer(cnn, normalization_mean, normalization_std,
                       content_img, style_img, input_img, style_mask, content_mask, device, 
                       lr=1.0, num_steps=300, style_weight=1000000, content_weight=1, sim_weight=10):
    """Run the style transfer."""
    print('Building the style transfer model..')
    model, style_losses, content_losses, sim_losses = get_style_model_and_losses(cnn,
        normalization_mean, normalization_std, style_img, content_img, style_mask, content_mask, device)
    optimizer = get_input_optimizer(input_img, lr)

    print('Optimizing..')
    run = [0]
    while run[0] <= num_steps:

        def closure():
            # correct the values of updated input image
            input_img.data.clamp_(0, 1)

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

            style_score *= style_weight
            content_score *= content_weight
            sim_score *= sim_weight

            loss = style_score + content_score + sim_score
            loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print('run {}: Style Loss : {:4f} Content Loss: {:4f} Similarity Loss: {:4f}'.format(
                    run, style_score.item(), content_score.item(), sim_score.item()))

            return style_score + content_score + sim_score

        optimizer.step(closure)

    # a last correction...
    input_img.data.clamp_(0, 1)

    return input_img
