import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
# import dlib
from transform import cifar_default_data_transforms
from PIL import Image as pil_image
import numpy as np
import math


class NRMSELoss(torch.nn.Module):
    def __init__(self):
        super(NRMSELoss,self).__init__()

    def forward(self,x,y):
        eps = 1e-6
        criterion = nn.MSELoss(reduction='mean')
        loss = torch.sqrt(criterion(x, y)+eps)
        return loss



def preprocess_image(image, data = 'train', cuda=False):
    """
    Preprocesses the image such that it can be fed into our network.
    During this process we envoke PIL to cast it into a PIL image.
    :param image: numpy image in opencv form (i.e., BGR and of shape
    :return: pytorch tensor of shape [1, 3, image_size, image_size], not
    necessarily casted to cuda
    """
    # Revert from BGR
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Preprocess using the preprocessing function used during training and
    # casting it to PIL image
    preprocess = cifar_default_data_transforms[data]

    preprocessed_image = preprocess(pil_image.fromarray(image))
    
    # Add first dimension as the network expects a batch
    #preprocessed_image = preprocessed_image.unsqueeze(0)
    if cuda:
        preprocessed_image = preprocessed_image.cuda()
    return preprocessed_image


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor




# PSNR Calculation
def psnr(img1, img2):
    img1_a = np.array(img1)
    img2_a = np.array(img2)
    mse = np.mean((img1_a - img2_a) ** 2)

    if mse == 0:
        return 100
    return 20*math.log10(255.0/math.sqrt(mse))













