# coding: utf-8
import cv2
from torch.utils.data import Dataset
import Transform_Model as TM
import random
# import dlib
import numpy as np
from PIL import Image
import torchvision.transforms.functional as tf
from torchvision import transforms
import torch

class FaceDataset(Dataset):
    def __init__(self, txt_path, transform = None):
        fh= open(txt_path, 'r')
        clean_imgs = []
        adv_imgs = []
        for line in fh:
            line = line.rstrip()
            words = line.split()
            clean_imgs.append(words[0])
            adv_imgs.append(words[1])

        self.clean_imgs = clean_imgs        # 最主要就是要生成这个list， 然后DataLoader中给index，通过getitem读取图片数据
        self.adv_imgs = adv_imgs
        self.transform = transform

    def rotation(self, image1, image2):

        # get a random angle range from (-180, 180)
        angle = transforms.RandomRotation.get_params([-180, 180])
        # same angle rotation for image1 and image2
        image1 = image1.rotate(angle)
        image2 = image2.rotate(angle)
        
        image1 = tf.to_tensor(image1)
        image2 = tf.to_tensor(image2)
        return image1, image2

    def flip(self, image1, image2):
        # 50% prob to horizontal flip and vertical flip
        if random.random() > 0.5:
            image1 = tf.hflip(image1)
            image2 = tf.hflip(image2)
        if random.random() > 0.5:
            image1 = tf.vflip(image1)
            image2 = tf.vflip(image2)
        image1 = tf.to_tensor(image1)
        image2 = tf.to_tensor(image2)
        return image1, image2
    def __getitem__(self, index):
        clean_address = self.clean_imgs[index]
        adv_address = self.adv_imgs[index]
        clean_img = TM.preprocess_image(cv2.imread(clean_address))
        
        
        adv_img = TM.preprocess_image(cv2.imread(adv_address))
        # if self.transform is not None:
  
        #     clean_img = self.transform(clean_img)
        #     adv_img = self.transform(adv_img)
        if self.transform == 'rotation':
            clean_img, adv_img = self.rotation(clean_img, adv_img)
        elif self.transform == 'flip':
            clean_img, adv_img = self.flip(clean_img, adv_img)
        else:
            clean_img = tf.to_tensor(clean_img)
            adv_img = tf.to_tensor(adv_img)
        return clean_img, adv_img


    def __len__(self):
        return len(self.clean_imgs)


class FaceDatasetTransformTest(Dataset):
    def __init__(self, txt_path, transform = None):
        fh= open(txt_path, 'r')
        clean_imgs = []
        adv_imgs = []
        for line in fh:
            line = line.rstrip()
            words = line.split()
            clean_imgs.append(words[0])
            adv_imgs.append(words[1])

        self.clean_imgs = clean_imgs        # 最主要就是要生成这个list， 然后DataLoader中给index，通过getitem读取图片数据
        self.adv_imgs = adv_imgs
        self.transform = transform

    def rotation(self, image1, image2):

        # get a random angle range from (-180, 180)
        angle = transforms.RandomRotation.get_params([-180, 180])
        # same angle rotation for image1 and image2
        image1 = image1.rotate(angle)
        image2 = image2.rotate(angle)
        
        image1 = tf.to_tensor(image1)
        image2 = tf.to_tensor(image2)
        return image1, image2

    def flip(self, image1, image2):
    # 50% prob to horizontal flip and vertical flip
        if random.random() > 0.5:
            image1 = tf.hflip(image1)
            image2 = tf.hflip(image2)
        if random.random() > 0.5:
            image1 = tf.vflip(image1)
            image2 = tf.vflip(image2)
        image1 = tf.to_tensor(image1)
        image2 = tf.to_tensor(image2)
        return image1, image2
    
    def hflip(self, image1, image2):
        image1 = tf.hflip(image1)
        image2 = tf.hflip(image2)
        image1 = tf.to_tensor(image1)
        image2 = tf.to_tensor(image2)
        return image1, image2
    
    def vflip(self, image1, image2):
        image1 = tf.vflip(image1)
        image2 = tf.vflip(image2)
        image1 = tf.to_tensor(image1)
        image2 = tf.to_tensor(image2)
        return image1, image2
    
    def rotation_new(self, image1, image2):

        
        if random.random() > 0.5:
            angle = transforms.RandomRotation.get_params([40, 50])
        else:
            angle = transforms.RandomRotation.get_params([-50, -40])
       
        image1 = image1.rotate(angle)
        image2 = image2.rotate(angle)
        
        image1 = tf.to_tensor(image1)
        image2 = tf.to_tensor(image2)
        return image1, image2

    def __getitem__(self, index):
        clean_address = self.clean_imgs[index]
        adv_address = self.adv_imgs[index]
        clean_img = TM.preprocess_image(cv2.imread(clean_address))
        
        
        adv_img = TM.preprocess_image(cv2.imread(adv_address))
        # if self.transform is not None:
  
        #     clean_img = self.transform(clean_img)
        #     adv_img = self.transform(adv_img)
        if self.transform == 'rotation':
            clean_img_transform, adv_img_transform = self.rotation(clean_img, adv_img)
        elif self.transform == 'flip':
            clean_img_transform, adv_img_transform = self.flip(clean_img, adv_img)
        elif self.transform == 'hflip':
            clean_img_transform, adv_img_transform = self.hflip(clean_img, adv_img)
        elif self.transform == 'vflip':
            clean_img_transform, adv_img_transform = self.vflip(clean_img, adv_img)
        elif self.transform == 'rotation_new':
            clean_img_transform, adv_img_transform = self.rotation_new(clean_img, adv_img)
        clean_img = tf.to_tensor(clean_img)
        adv_img = tf.to_tensor(adv_img)
        return clean_img, adv_img, clean_img_transform, adv_img_transform


    def __len__(self):
        return len(self.clean_imgs)

class Labeled_FaceDataset(Dataset):
    def __init__(self, txt_path, label):
        fh = open(txt_path, 'r')
        clean_imgs = []
        adv_imgs = []
        # labels = []
        for line in fh:
            line = line.rstrip()
            words = line.split()
            clean_imgs.append(words[0])
            adv_imgs.append(words[1])
            # labels.append(label)

        self.clean_imgs = clean_imgs        # 最主要就是要生成这个list， 然后DataLoader中给index，通过getitem读取图片数据
        self.adv_imgs = adv_imgs
        self.label = label

    def __getitem__(self, index):
        clean_address = self.clean_imgs[index]
        adv_address = self.adv_imgs[index]
        clean_img = TM.preprocess_image(cv2.imread(clean_address))
        adv_img = TM.preprocess_image(cv2.imread(adv_address))
        # print(clean_img.type)
        clean_img = tf.to_tensor(clean_img)
        adv_img = tf.to_tensor(adv_img)
        return torch.cat((adv_img-clean_img, clean_img),0), self.label

    def __len__(self):
        return len(self.clean_imgs)

class Labeled_FaceDataset_new(Dataset):
    def __init__(self, txt_path, label):
        fh = open(txt_path, 'r')
        clean_imgs = []
        adv_imgs = []
        # labels = []
        for line in fh:
            line = line.rstrip()
            words = line.split()
            clean_imgs.append(words[0])
            adv_imgs.append(words[1])
            # labels.append(label)

        self.clean_imgs = clean_imgs        # 最主要就是要生成这个list， 然后DataLoader中给index，通过getitem读取图片数据
        self.adv_imgs = adv_imgs
        self.label = label

    def __getitem__(self, index):
        clean_address = self.clean_imgs[index]
        adv_address = self.adv_imgs[index]
        clean_img = TM.preprocess_image(cv2.imread(clean_address))
        adv_img = TM.preprocess_image(cv2.imread(adv_address))
        # print(clean_img.type)
        clean_img = tf.to_tensor(clean_img)
        adv_img = tf.to_tensor(adv_img)
        return (adv_img - clean_img), self.label

    def __len__(self):
        return len(self.clean_imgs)

class Labeled_FaceDataset_incremental(Dataset):
    def __init__(self, txt_path, label, known):
        fh = open(txt_path, 'r')
        clean_imgs = []
        adv_imgs = []
        # labels = []
        for line in fh:
            line = line.rstrip()
            words = line.split()
            clean_imgs.append(words[0])
            adv_imgs.append(words[1])
            # labels.append(label)

        self.clean_imgs = clean_imgs        # 最主要就是要生成这个list， 然后DataLoader中给index，通过getitem读取图片数据
        self.adv_imgs = adv_imgs
        self.label = label
        self.known = known

    def __getitem__(self, index):
        clean_address = self.clean_imgs[index]
        adv_address = self.adv_imgs[index]
        clean_img = TM.preprocess_image(cv2.imread(clean_address))
        adv_img = TM.preprocess_image(cv2.imread(adv_address))
        # print(clean_img.type)
        clean_img = tf.to_tensor(clean_img)
        adv_img = tf.to_tensor(adv_img)
        return (adv_img - clean_img), self.label, self.known

    def __len__(self):
        return len(self.clean_imgs)