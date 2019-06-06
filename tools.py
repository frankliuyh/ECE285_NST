import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as td
import torchvision as tv
from PIL import Image
import matplotlib.pyplot as plt

def myimshow(image, ax=plt):
    img = image.cpu().clone().clamp(0, 255).numpy()
    img = img.transpose(1, 2, 0).astype("uint8")
    img = Image.fromarray(img)
    h = ax.imshow(img)
    ax.axis('off')
    return h

def load_image(path, size, device):
    img = Image.open(path)
    img = img.resize(size, Image.ANTIALIAS)
    transform = tv.transforms.Compose([
            tv.transforms.ToTensor(),
            tv.transforms.Lambda(lambda x: x.mul(255))
            ])
    img = transform(img)
    img = img.reshape((1, 3, size[0], size[1]))
    img = img.to(device)
    return img

def stylize(content_path, style_path, transformer_path, size, device):
    content = load_image(content_path, size, device)
    style = load_image(style_path, size, device)
    transformer = torch.load(transformer_path, map_location = device).eval()
    with torch.no_grad():
        output = transformer(content)
    fig, axes = plt.subplots(ncols = 3, figsize = (16, 16))
    myimshow(content[0], axes[0])
    axes[0].set_title("Content image")
    myimshow(output[0], axes[1])
    axes[1].set_title("Output image")
    myimshow(style[0], axes[2])
    axes[2].set_title("Style image")