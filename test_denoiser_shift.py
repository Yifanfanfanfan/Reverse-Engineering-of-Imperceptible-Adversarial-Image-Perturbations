# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from architectures import get_architecture, IMAGENET_CLASSIFIERS
from datasets import get_dataset, DATASETS
from torch.nn import MSELoss, CrossEntropyLoss, L1Loss
from torch.optim import SGD, Optimizer, Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage
from train_utils import AverageMeter, accuracy, init_logfile, log
from scipy.stats import spearmanr, kendalltau
import torch.utils.data as data
from torchvision import models
from torchvision import transforms
import argparse
from datetime import datetime
import numpy as np
import os
import time
import torch
import torch.nn as nn
import dataset_txt_generate as tt
import RED_Dataset as RD
import matplotlib.pyplot as plt
import random
import torchvision.transforms.functional as tf
from torchvision import transforms
import math
from collections import OrderedDict

# import heapq
# import test_utils

toPilImage = ToPILImage()

def main(args):
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    results = {}
    args.denoiser = os.path.join(os.getenv('PT_DATA_DIR', './'), args.denoiser)
    checkpoint = torch.load(args.denoiser)
    denoiser = get_architecture(checkpoint['arch'] ,args.dataset)
    # if torch.cuda.is_available():
    #     denoiser = torch.nn.DataParallel(denoiser).cuda()
    
    denoiser.load_state_dict(checkpoint['state_dict'])
    print(denoiser)
    denoiser.cuda().eval()

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    args.outdir = os.path.join(os.getenv('PT_OUTPUT_DIR', './'), args.outdir)

    class Normalize(nn.Module):
        def __init__(self, mean, std):
            super(Normalize, self).__init__()
            self.register_buffer('mean', torch.Tensor(mean))
            self.register_buffer('std', torch.Tensor(std))

        def forward(self, input):
            # Broadcasting
            mean = self.mean.reshape(1, 3, 1, 1)
            std = self.std.reshape(1, 3, 1, 1)
            return (input - mean) / std
        
    norm_layer = Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    
    root = args.root
    attack_method = list(map(lambda x: str(x), args.attack_method.split(",")))
    victim_model = list(map(lambda x: str(x), args.victim_model.split(",")))
    intensity = list(map(lambda x: str(x), args.intensity.split(",")))

    # clf_losses_denoised_advs = np.zeros((len(attack_method), len(victim_model)))  #D(x') x' L2
    # clf_losses_cleans_advs =  np.zeros((len(attack_method), len(victim_model)))  # x' x  L2
    # clf_losses_denoised_cleans = np.zeros((len(attack_method), len(victim_model))) # D(x') x L2
    # clf_losses_denoised_advs_L1 = np.zeros((len(attack_method), len(victim_model)))  #D(x') x' L1
    # clf_losses_cleans_advs_L1 = np.zeros((len(attack_method), len(victim_model)))  # x' x  L1
    # clf_losses_denoised_cleans_L1 = np.zeros((len(attack_method), len(victim_model))) # D(x') x  L1
    image_num = np.zeros((len(attack_method), len(victim_model)))
    clf_acc = np.zeros((len(attack_method), len(victim_model)))
    clf_acc_perturb = np.zeros((len(attack_method), len(victim_model)))



    accs_adv_a = np.zeros((21, len(attack_method), len(victim_model)))
    suc_rates_adv_a = np.zeros((21, len(attack_method), len(victim_model)))

    acc_flip = np.zeros((len(attack_method), len(victim_model)))
    suc_rate_flip = np.zeros((len(attack_method), len(victim_model)))
    acc_rotation = np.zeros((len(attack_method), len(victim_model)))
    suc_rate_rotation = np.zeros((len(attack_method), len(victim_model)))

    distance_loss_denoised_cleans = np.zeros((len(attack_method), len(victim_model)))
    distance_loss_cleans_advs = np.zeros((len(attack_method), len(victim_model)))
    distance_loss_denoised_advs = np.zeros((len(attack_method), len(victim_model)))

    logit_distance_loss_denoised_cleans = np.zeros((len(attack_method), len(victim_model)))
    logit_distance_loss_cleans_advs = np.zeros((len(attack_method), len(victim_model)))
    logit_distance_loss_denoised_advs = np.zeros((len(attack_method), len(victim_model)))
    logit_distance_loss_recon_advs = np.zeros((len(attack_method), len(victim_model)))
    logit_distance_loss_recon_cleans = np.zeros((len(attack_method), len(victim_model)))
    
    if intensity[0] == 'auto':
        text_test_list = os.path.join(root, 'test_list.txt')
        img_folder = args.img_folder + victim_model[0]
        img_test_clean = img_folder + '/clean'
        img_test_adv = img_folder + '/adv'
        tt.gen_txt_auto(text_test_list, img_test_clean, img_test_adv)
        test_data = RD.FaceDataset(text_test_list)
        clf_gt = nn.Sequential(norm_layer,models.resnet50(pretrained=True)).cuda().eval()
        clf = clf_gt
        test_loader = DataLoader(dataset=test_data, batch_size=args.batch, num_workers=args.workers, pin_memory=False)
        classification_criterion = CrossEntropyLoss(size_average=None, reduce=None, reduction = 'mean').cuda()

        # clf_losses_denoised_advs_auto, clf_losses_cleans_advs_auto, clf_losses_denoised_cleans_auto,\
        # clf_losses_denoised_advs_L1_auto, clf_losses_cleans_advs_L1_auto, clf_losses_denoised_cleans_L1_auto, \

        clf_acc_auto, clf_acc_perturb_auto, image_num_auto, \
        distance_loss_denoised_cleans_auto, distance_loss_cleans_advs_auto, distance_loss_denoised_advs_auto, \
        logit_distance_loss_denoised_cleans_auto, logit_distance_loss_cleans_advs_auto, logit_distance_loss_denoised_advs_auto\
        = test_with_classifier(test_loader, denoiser, classification_criterion, args.noise_sd, args.print_freq, clf,clf_gt, None, None)
        print('The average clean accuracy and attack successful rate across the dataset is {}, {}'.format(clf_acc_auto,clf_acc_perturb_auto))
        print('The distance beween clean and denoised is {}'.format(distance_loss_denoised_cleans_auto))
        print('The distance beween clean and adv is {}'.format(distance_loss_cleans_advs_auto))
        print('The distance between adv and denoised is {}'.format(distance_loss_denoised_advs_auto))
        print('The logit distance beween clean and denoised is {}'.format(logit_distance_loss_denoised_cleans_auto))
        print('The logit distance beween clean and adv is {}'.format(logit_distance_loss_cleans_advs_auto))
        print('The logit distance between adv and denoised is {}'.format(logit_distance_loss_denoised_advs_auto))
        
        # print('The average MSE between denoised images and clean images across the dataset is {}'.format(clf_losses_denoised_cleans_auto))
        # print('The average MSE between clean images and adv images across the dataset is {}'.format(clf_losses_cleans_advs_auto))
        # print('The average MSE between denoised images and adv images across the dataset is {}'.format(clf_losses_denoised_advs_auto))
        # print('The average MAE between denoised images and clean images across the dataset is {}'.format(clf_losses_denoised_cleans_L1_auto))
        # print('The average MAE between clean images and adv images across the dataset is {}'.format(clf_losses_cleans_advs_L1_auto))
        # print('The average MAE between denoised images and adv images across the dataset is {}'.format(clf_losses_denoised_advs_L1_auto))

    elif intensity[0] == 'robust_auto':
        text_test_list = os.path.join(root, 'test_list.txt')
        img_folder = args.img_folder
        img_test_clean = img_folder + '/clean'
        img_test_adv = img_folder + '/adv'
        tt.gen_txt_robust_auto(text_test_list, img_test_clean, img_test_adv)
        test_data = RD.FaceDataset(text_test_list)
        vic_checkpoint = torch.load(args.robust_res50_path)
        clf = models.__dict__['resnet50']()

        state_dict = vic_checkpoint['state_dict']
        new_state_dict = OrderedDict()

        for k, v in state_dict.items():
            k = k.replace('module.', '')
            new_state_dict[k] = v
        
        clf.load_state_dict(new_state_dict)
        clf = clf.cuda().eval()

        clf_gt = clf
        test_loader = DataLoader(dataset=test_data, batch_size=args.batch, num_workers=args.workers, pin_memory=False)
        classification_criterion = CrossEntropyLoss(size_average=None, reduce=None, reduction = 'mean').cuda()
        # clf_losses_denoised_advs_auto, clf_losses_cleans_advs_auto, clf_losses_denoised_cleans_auto, \
        # clf_losses_denoised_advs_L1_auto, clf_losses_cleans_advs_L1_auto, clf_losses_denoised_cleans_L1_auto, \
        clf_acc_auto, clf_acc_perturb_auto, image_num_auto,\
        distance_loss_denoised_cleans_auto, distance_loss_cleans_advs_auto, distance_loss_denoised_advs_auto,\
        logit_distance_loss_denoised_cleans_auto, logit_distance_loss_cleans_advs_auto, logit_distance_loss_denoised_advs_auto\
        = test_with_classifier(test_loader, denoiser, classification_criterion, args.noise_sd, args.print_freq, clf,clf_gt, None, None)
        print('The average clean accuracy and attack successful rate across the dataset is {}, {}'.format(clf_acc_auto,clf_acc_perturb_auto))
        print('The distance beween clean and denoised is {}'.format(distance_loss_denoised_cleans_auto))
        print('The distance beween clean and adv is {}'.format(distance_loss_cleans_advs_auto))
        print('The distance between adv and denoised is {}'.format(distance_loss_denoised_advs_auto))
        print('The logit distance beween clean and denoised is {}'.format(logit_distance_loss_denoised_cleans_auto))
        print('The logit distance beween clean and adv is {}'.format(logit_distance_loss_cleans_advs_auto))
        print('The logit distance between adv and denoised is {}'.format(logit_distance_loss_denoised_advs_auto))
        # print('The average MSE between denoised images and clean images across the dataset is {}'.format(clf_losses_denoised_cleans_auto))
        # print('The average MSE between clean images and adv images across the dataset is {}'.format(clf_losses_cleans_advs_auto))
        # print('The average MSE between denoised images and adv images across the dataset is {}'.format(clf_losses_denoised_advs_auto))
        # print('The average MAE between denoised images and clean images across the dataset is {}'.format(clf_losses_denoised_cleans_L1_auto))
        # print('The average MAE between clean images and adv images across the dataset is {}'.format(clf_losses_cleans_advs_L1_auto))
        # print('The average MAE between denoised images and adv images across the dataset is {}'.format(clf_losses_denoised_advs_L1_auto))
    
    elif intensity[0] == 'feature':
        model_num = 0
        clf_acc_feature = np.zeros(len(victim_model))
        clf_acc_perturb_feature = np.zeros(len(victim_model))
        image_num_feature = np.zeros(len(victim_model))
        distance_loss_denoised_cleans_feature = np.zeros(len(victim_model))
        distance_loss_cleans_advs_feature = np.zeros(len(victim_model))
        distance_loss_denoised_advs_feature = np.zeros(len(victim_model))

        logit_distance_loss_denoised_cleans_feature = np.zeros(len(victim_model))
        logit_distance_loss_cleans_advs_feature = np.zeros(len(victim_model))
        logit_distance_loss_denoised_advs_feature = np.zeros(len(victim_model))
    
        for model_name in victim_model: #vgg19,alexnet
            
            text_test_list = os.path.join(root, 'text/test_list_{}_feature.txt'.format(model_name))
            img_folder = args.img_folder
            img_test_clean = img_folder + '/saved_images_' + model_name + '/Clean(Source)'
            img_test_adv = img_folder + '/saved_images_' + model_name + '/Adv'
            tt.gen_txt_auto(text_test_list, img_test_clean, img_test_adv)
            test_data = RD.FaceDataset(text_test_list)
            if model_name == "alexnet":
                clf_gt = nn.Sequential(norm_layer, models.alexnet(pretrained=True)).cuda().eval()
            elif model_name == "vgg19":
                clf_gt = nn.Sequential(norm_layer, models.vgg19(pretrained=True)).cuda().eval()
            clf = clf_gt
            classification_criterion = CrossEntropyLoss(size_average=None, reduce=None, reduction = 'mean').cuda()
            test_loader = DataLoader(dataset=test_data, batch_size=args.batch, num_workers=args.workers, pin_memory=False)
            clf_acc_feature[model_num], clf_acc_perturb_feature[model_num], image_num_feature[model_num],\
                    distance_loss_denoised_cleans_feature[model_num], distance_loss_cleans_advs_feature[model_num], distance_loss_denoised_advs_feature[model_num],\
                    logit_distance_loss_denoised_cleans_feature[model_num], logit_distance_loss_cleans_advs_feature[model_num], logit_distance_loss_denoised_advs_feature[model_num]\
                    = test_with_classifier(test_loader, denoiser, classification_criterion, args.noise_sd, args.print_freq, clf, clf_gt, model_name, None)
            model_num = model_num + 1
        print('clf_acc is')
        print(clf_acc_feature)
        print('success_rate is')
        print(clf_acc_perturb_feature)

        clf_acc_avg = np.mean(clf_acc_feature)  
        clf_acc_perturb_avg = np.mean(clf_acc_perturb_feature) 
        distance_loss_denoised_cleans_avg = np.sum(distance_loss_denoised_cleans_feature*image_num_feature)/np.sum(image_num_feature)
        distance_loss_cleans_advs_avg = np.sum(distance_loss_cleans_advs_feature*image_num_feature)/np.sum(image_num_feature)
        distance_loss_denoised_advs_avg = np.sum(distance_loss_denoised_advs_feature*image_num_feature)/np.sum(image_num_feature)
        logit_distance_loss_denoised_advs_avg = np.sum(logit_distance_loss_denoised_advs_feature*image_num_feature)/np.sum(image_num_feature)
        logit_distance_loss_cleans_advs_avg = np.sum(logit_distance_loss_cleans_advs_feature*image_num_feature)/np.sum(image_num_feature)
        logit_distance_loss_denoised_cleans_avg = np.sum(logit_distance_loss_denoised_cleans_feature*image_num_feature)/np.sum(image_num_feature)
        print('The average clean accuracy and attack successful rate across the dataset is {}, {}'.format(clf_acc_avg,clf_acc_perturb_avg))
        print('The distance beween clean and denoised is {}'.format(distance_loss_denoised_cleans_avg))
        print('The distance beween clean and adv is {}'.format(distance_loss_cleans_advs_avg))
        print('The distance between adv and denoised is {}'.format(distance_loss_denoised_advs_avg))
        print('The logit distance beween clean and denoised is {}'.format(logit_distance_loss_denoised_cleans_avg))
        print('The logit distance beween clean and adv is {}'.format(logit_distance_loss_cleans_advs_avg))
        print('The logit distance between adv and denoised is {}'.format(logit_distance_loss_denoised_advs_avg))

    else:
        model_num = 0
        # transform_set = ['flip', 'rotation']
        for model_name in victim_model: #res18,res50,vgg16,vgg19,incptv3

            attack_num = 0
            for attack_name in attack_method: #PGD,FGSM,CW
                test_data_list = list()
                for intensity_name in intensity:
                    text_test_list = os.path.join(root, 'test_list_portion_{}_{}_{}.txt'.format(model_name, attack_name, intensity_name))
                    img_folder = args.img_folder
                    # img_test_clean = '/home/yifan/github/RED_Denoising/RED_Denoising/code/denoise/' + model_name + '/testclean/' + args.intensity
                    # img_test_adv = '/home/yifan/github/RED_Denoising/RED_Denoising/code/denoise/' + model_name + '/test/' + attack_name + '/' + args.intensity  
                    img_test_clean = img_folder + attack_name + model_name + '/test4types/clean'
                    img_test_adv = img_folder + attack_name + model_name + '/test4types/' + intensity_name
                    tt.gen_txt_new_new('test_list_portion_{}_{}_{}.txt'.format(model_name, attack_name, intensity_name), img_test_clean, img_test_adv)
                    test_data_method_model = RD.FaceDataset(text_test_list)


                    test_data_list.append(RD.FaceDataset(text_test_list))
                    test_data_method_model = data.ConcatDataset(test_data_list)

                if model_name == "incptv3":
                    clf_gt = nn.Sequential(norm_layer,models.inception_v3(pretrained=True)).cuda().eval()
                    # print(clf_gt)
                elif model_name == "vgg16":
                    clf_gt = nn.Sequential(norm_layer, models.vgg16(pretrained=True)).cuda().eval()
                elif model_name == "vgg19":
                    clf_gt = nn.Sequential(norm_layer, models.vgg19(pretrained=True)).cuda().eval()
                elif model_name == "res18":
                    clf_gt = nn.Sequential(norm_layer,models.resnet18(pretrained=True)).cuda().eval()
                elif model_name == "res50":
                    clf_gt = nn.Sequential(norm_layer,models.resnet50(pretrained=True)).cuda().eval()
                clf = clf_gt

                classification_criterion = CrossEntropyLoss(size_average=None, reduce=None, reduction = 'mean').cuda()
                test_loader = DataLoader(dataset=test_data_method_model, batch_size=args.batch, num_workers=args.workers, pin_memory=False)

                
                if args.test_DTx == 1:
                    acc_flip[attack_num,model_num], suc_rate_flip[attack_num,model_num], acc_rotation[attack_num,model_num], suc_rate_rotation[attack_num,model_num] = test_with_classifier_DTx(test_loader, denoiser, classification_criterion, args.noise_sd, args.print_freq, clf,clf_gt)

                elif args.test_TDx == 1:
                    acc_flip[attack_num,model_num], suc_rate_flip[attack_num,model_num], acc_rotation[attack_num,model_num], suc_rate_rotation[attack_num,model_num] = test_with_classifier_TDx(test_loader, denoiser, classification_criterion, args.noise_sd, args.print_freq, clf,clf_gt)
                else:
                    # clf_losses_denoised_advs[attack_num,model_num], clf_losses_cleans_advs[attack_num,model_num], \
                    # clf_losses_denoised_cleans[attack_num,model_num], clf_losses_denoised_advs_L1[attack_num,model_num],\
                    # clf_losses_cleans_advs_L1[attack_num,model_num], clf_losses_denoised_cleans_L1[attack_num,model_num],\
                    clf_acc[attack_num,model_num], clf_acc_perturb[attack_num,model_num], image_num[attack_num,model_num],\
                    distance_loss_denoised_cleans[attack_num, model_num], distance_loss_cleans_advs[attack_num, model_num], distance_loss_denoised_advs[attack_num, model_num],\
                    logit_distance_loss_denoised_cleans[attack_num, model_num], logit_distance_loss_cleans_advs[attack_num, model_num], logit_distance_loss_denoised_advs[attack_num, model_num],\
                    logit_distance_loss_recon_advs[attack_num, model_num], logit_distance_loss_recon_cleans[attack_num, model_num]\
                    = test_with_classifier(test_loader, denoiser, classification_criterion, args.noise_sd, args.print_freq, clf,clf_gt, model_name, attack_name)

                # print('The average MSE between denoised images and clean images for attack method {} and victim model {} is {}'.format(attack_name,model_name,clf_losses_denoised_cleans[attack_num,model_num]))
                # print('The average MSE between clean images and adv images for attack method {} and victim model {} is {}'.format(attack_name,model_name,clf_losses_cleans_advs[attack_num,model_num]))
                # print('The average MSE between denoised images and adv images for attack method {} and victim model {} is {}'.format(attack_name,model_name,clf_losses_denoised_advs[attack_num,model_num]))
                # print('The average MAE between denoised images and clean images for attack method {} and victim model {} is {}'.format(attack_name,model_name,clf_losses_denoised_cleans_L1[attack_num,model_num]))
                # print('The average MAE between clean images and adv images for attack method {} and victim model {} is {}'.format(attack_name,model_name,clf_losses_cleans_advs_L1[attack_num,model_num]))
                # print('The average MAE between denoised images and adv images for attack method {} and victim model {} is {}'.format(attack_name,model_name,clf_losses_denoised_advs_L1[attack_num,model_num]))
                # print('Reconstructed Image Accuracy for attack method {} and victim model {} is {}'.format(attack_name,model_name,clf_acc[attack_num,model_num]))
                # print('Reconstructed Adversarial Image Accuracy (compared with the label of the adversarial example) for attack method {} and victim model {} is {}'.format(attack_name,model_name,clf_acc_perturb[attack_num,model_num]))
                

                attack_num = attack_num + 1
            model_num = model_num + 1

        if (args.test_DTx == 1) or (args.test_TDx == 1):
            print('acc_flip is')
            print(acc_flip)
            print('suc_rate_flip is')
            print(suc_rate_flip)
            print('acc_rotation is')
            print(acc_rotation)
            print('suc_rate_rotation is')
            print(suc_rate_rotation)
            print('The avg acc_flip across dataset is {}'.format(np.mean(acc_flip)))
            print('The avg suc_rate_flip across dataset is {}'.format(np.mean(suc_rate_flip)))
            print('The avg acc_rotation across dataset is {}'.format(np.mean(acc_rotation)))
            print('The avg suc_rate_rotation across dataset is {}'.format(np.mean(suc_rate_rotation)))
        else:
            print('clf_acc is')
            print(clf_acc)
            print('success_rate is')
            print(clf_acc_perturb)
            # clf_losses_cleans_advs_avg = np.sum(clf_losses_cleans_advs*image_num)/np.sum(image_num) 
            # clf_losses_denoised_advs_avg = np.sum(clf_losses_denoised_advs*image_num)/np.sum(image_num) 
            # clf_losses_denoised_cleans_avg =  np.sum(clf_losses_denoised_cleans*image_num)/np.sum(image_num) 
            # clf_losses_cleans_advs_L1_avg = np.sum(clf_losses_cleans_advs_L1* image_num)/np.sum(image_num) 
            # clf_losses_denoised_advs_L1_avg = np.sum(clf_losses_denoised_advs_L1*image_num)/np.sum(image_num)
            # clf_losses_denoised_cleans_L1_avg = np.sum(clf_losses_denoised_cleans_L1*image_num)/np.sum(image_num) 
            # clf_acc_avg = np.sum(clf_acc*image_num)/np.sum(image_num)  
            # clf_acc_perturb_avg = np.sum(clf_acc_perturb*image_num)/np.sum(image_num) 

            distance_loss_denoised_cleans_avg = np.sum(distance_loss_denoised_cleans*image_num)/np.sum(image_num)
            distance_loss_cleans_advs_avg = np.sum(distance_loss_cleans_advs*image_num)/np.sum(image_num)
            distance_loss_denoised_advs_avg = np.sum(distance_loss_denoised_advs*image_num)/np.sum(image_num)
            logit_distance_loss_denoised_advs_avg = np.sum(logit_distance_loss_denoised_advs*image_num)/np.sum(image_num)
            logit_distance_loss_cleans_advs_avg = np.sum(logit_distance_loss_cleans_advs*image_num)/np.sum(image_num)
            logit_distance_loss_denoised_cleans_avg = np.sum(logit_distance_loss_denoised_cleans*image_num)/np.sum(image_num)
            logit_distance_loss_recon_advs_avg = np.sum(logit_distance_loss_recon_advs*image_num)/np.sum(image_num)
            logit_distance_loss_recon_cleans_avg = np.sum(logit_distance_loss_recon_cleans*image_num)/np.sum(image_num)

            # np.savez("perturbation_interpolation_test/image_num.npy", image_num)
            # clf_losses_cleans_advs_avg = np.mean(clf_losses_cleans_advs)
            # clf_losses_denoised_advs_avg = np.mean(clf_losses_denoised_advs)
            # clf_losses_denoised_cleans_avg =  np.mean(clf_losses_denoised_cleans)
            # clf_losses_cleans_advs_L1_avg = np.mean(clf_losses_cleans_advs_L1) 
            # clf_losses_denoised_advs_L1_avg = np.mean(clf_losses_denoised_advs_L1)
            # clf_losses_denoised_cleans_L1_avg = np.mean(clf_losses_denoised_cleans_L1) 
            clf_acc_avg = np.mean(clf_acc)  
            clf_acc_perturb_avg = np.mean(clf_acc_perturb) 
            
            print('The average clean accuracy and attack successful rate across the dataset is {}, {}'.format(clf_acc_avg,clf_acc_perturb_avg))
            # print('The average MSE between denoised images and clean images across the dataset is {}'.format(clf_losses_denoised_cleans_avg))
            # print('The average MSE between clean images and adv images across the dataset is {}'.format(clf_losses_cleans_advs_avg))
            # print('The average MSE between denoised images and adv images across the dataset is {}'.format(clf_losses_denoised_advs_avg))
            # print('The average MAE between denoised images and clean images across the dataset is {}'.format(clf_losses_denoised_cleans_L1_avg))
            # print('The average MAE between clean images and adv images across the dataset is {}'.format(clf_losses_cleans_advs_L1_avg))
            # print('The average MAE between denoised images and adv images across the dataset is {}'.format(clf_losses_denoised_advs_L1_avg))
            print('The distance beween clean and denoised is {}'.format(distance_loss_denoised_cleans_avg))
            print('The distance beween clean and adv is {}'.format(distance_loss_cleans_advs_avg))
            print('The distance between adv and denoised is {}'.format(distance_loss_denoised_advs_avg))
            print('The logit distance beween clean and denoised is {}'.format(logit_distance_loss_denoised_cleans_avg))
            print('The logit distance beween clean and adv is {}'.format(logit_distance_loss_cleans_advs_avg))
            print('The logit distance between adv and denoised is {}'.format(logit_distance_loss_denoised_advs_avg))
            print('The logit distance between recon and adv is {}'.format(logit_distance_loss_recon_advs_avg))
            print('The logit distance between recon and clean is {}'.format(logit_distance_loss_recon_cleans_avg))

            
        # clf_loss_attack_method = np.mean(clf_losses_denoised_cleans,axis=1)
        # clf_acc_attack_method = np.mean(clf_acc,axis=1)
        # clf_acc_perturb_attack_method = np.mean(clf_acc_perturb, axis=1)
        # print('The average MSE, clean accuracy, and attack successful rate of reconstructed image for attacks are {}, {}, {}'.format(clf_loss_attack_method,clf_acc_attack_method, clf_acc_perturb_attack_method))

        # clf_loss_victim_model = np.mean(clf_losses_denoised_cleans,axis=0)
        # clf_acc_victim_model = np.mean(clf_acc,axis=0)
        # clf_acc_perturb_victim_model = np.mean(clf_acc_perturb, axis=0)
        # print('The average MSE, clean accuracy, and attack successful rate of reconstructed image for victim models are {}, {}, {}'.format(clf_loss_victim_model,clf_acc_victim_model, clf_acc_perturb_victim_model))
    
        
        # return clf_losses_denoised_advs_avg, clf_losses_denoised_cleans_avg, clf_losses_cleans_advs_avg, clf_acc, clf_acc_perturb
    return None
    

    



def test_with_classifier(loader: DataLoader, denoiser: torch.nn.Module, criterion, noise_sd: float, print_freq: int, classifier: torch.nn.Module, classifier_gt: torch.nn.Module, model_name, attack_name):
    """
    A function to test the classification performance of a denoiser when attached to a given classifier
        :param loader:DataLoader: test dataloader
        :param denoiser:torch.nn.Module: the denoiser 
        :param criterion: the loss function (e.g. CE)
        :param noise_sd:float: the std-dev of the Guassian noise perturbation of the input
        :param print_freq:int: the frequency of logging
        :param classifier:torch.nn.Module: the classifier to which the denoiser is attached
    """
    root = ''
    MSE = MSELoss(size_average=None, reduce=None, reduction='mean').cuda()
    L1 = L1Loss(size_average=None, reduce=None, reduction='mean').cuda()
    MSE_dis = MSELoss(size_average=None, reduce=None, reduction='sum').cuda()
    # log
    result_dir = os.path.join(root, "Result_test")

    now_time = datetime.now()
    time_str = datetime.strftime(now_time, '%m-%d_%H-%M-%S')

    log_dir = os.path.join(result_dir, time_str)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    outputFile = os.path.join(log_dir, 'log.txt')

    batch_time = AverageMeter()
    data_time = AverageMeter()
    # losses = AverageMeter()
    losses_denoised_advs = AverageMeter()  #D(x') x' L2
    losses_cleans_advs =  AverageMeter()  # x' x  L2
    losses_denoised_cleans = AverageMeter() # D(x') x L2
    losses_denoised_advs_L1 = AverageMeter()  #D(x') x' L1
    losses_cleans_advs_L1 =  AverageMeter()  # x' x  L1
    losses_denoised_cleans_L1 = AverageMeter() # D(x') x  L1
    top1 = AverageMeter()
    top1_perturb = AverageMeter()

    distance_losses_denoised_advs = AverageMeter()  #D(x') x' L2
    distance_losses_cleans_advs =  AverageMeter()  # x' x  L2
    distance_losses_denoised_cleans = AverageMeter() # D(x') x L2

    logit_distance_losses_denoised_advs = AverageMeter()  #D(x') x' L2
    logit_distance_losses_cleans_advs =  AverageMeter()  # x' x  L2
    logit_distance_losses_denoised_cleans = AverageMeter() # D(x') x L2
    logit_distance_losses_recon_advs = AverageMeter() 
    logit_distance_losses_recon_cleans = AverageMeter()


    # angles_adv = AverageMeter()
    # angles_clean = AverageMeter()
    # angles_logit_clean_denoised = AverageMeter()
    # angles_logit_adv_recon = AverageMeter()
    # angles_logit_adv_recon = AverageMeter()
    # angles_logit_clean_recon = AverageMeter()

    end = time.time()

    # switch to eval mode
    classifier.eval()
    classifier_gt.eval()
    if denoiser:
        denoiser.eval()

    with torch.no_grad():
        angle_dir = 'angle_all/angle_mu{}_gamma{}_lambda{}_data{}/'.format(args.mu1, args.gamma1, args.lambda1, args.reg_data)
        distance_dir = 'distance/distance_mu{}_gamma{}_lambda{}_data{}/'.format(args.mu1, args.gamma1, args.lambda1, args.reg_data)
        if not os.path.exists(angle_dir):
            os.makedirs(angle_dir)
        if not os.path.exists(distance_dir):
            os.makedirs(distance_dir)
        if not os.path.exists(angle_dir+'adv/'):
            os.makedirs(angle_dir+'adv/')
        if not os.path.exists(angle_dir+'clean/'):
            os.makedirs(angle_dir+'clean/')
        if not os.path.exists(angle_dir+'logit_clean_denoised/'):
            os.makedirs(angle_dir+'logit_clean_denoised/')
        if not os.path.exists(angle_dir+'logit_adv_denoised/'):
            os.makedirs(angle_dir+'logit_adv_denoised/')
        if not os.path.exists(angle_dir+'logit_clean_recon/'):
            os.makedirs(angle_dir+'logit_clean_recon/')
        if not os.path.exists(angle_dir+'logit_adv_recon/'):
            os.makedirs(angle_dir+'logit_adv_recon/')

        image_num = 0
        for i, (cleans, advs) in enumerate(loader):
            # print(cleans.shape)
            image_num = image_num + cleans.shape[0]
            # measure data loading time
            data_time.update(time.time() - end)

            cleans = cleans.cuda().to(dtype=torch.float)
            advs = advs.cuda().to(dtype=torch.float) 
            perturbation = advs - cleans
            
            if denoiser is not None:
                denoised = denoiser(advs)
            
            f_recon = classifier(advs - denoised + cleans)
            f_denoised = classifier(denoised)
            f_clean = classifier(cleans)
            f_adv = classifier(advs)

            F_recon = f_recon.argmax(1).detach().clone()
            
            F_denoised = f_denoised.argmax(1).detach().clone()

            F_clean = f_clean.argmax(1).detach().clone()

            F_adv = f_adv.argmax(1).detach().clone()
            
            loss_denoised_advs_L1 = L1(denoised, advs)
            loss_cleans_advs_L1 = L1(cleans, advs)
            loss_denoised_cleans_L1 = L1(denoised, cleans)
            loss_denoised_advs = MSE(denoised, advs)
            loss_cleans_advs = MSE(cleans, advs)
            loss_denoised_cleans = MSE(denoised, cleans)
            acc_denoised = sum(F_denoised==F_clean)/cleans.size(0)
            acc_perturb = sum(F_recon==F_adv)/cleans.size(0)
            
            # -----------------distance calculation start--------------------
           
            distance_loss_denoised_cleans = torch.mean(torch.norm(denoised-cleans, p=2, dim=[1,2,3]))
            distance_loss_denoised_advs = torch.mean(torch.norm(denoised-advs, p=2, dim=[1,2,3]))
            distance_loss_cleans_advs = torch.mean(torch.norm(cleans-advs, p=2, dim=[1,2,3]))
            logit_distance_loss_denoised_advs = torch.mean(torch.norm(f_denoised-f_adv, p=2, dim=1))
            logit_distance_loss_cleans_advs = torch.mean(torch.norm(f_clean-f_adv, p=2, dim=1))
            logit_distance_loss_denoised_cleans = torch.mean(torch.norm(f_denoised-f_clean, p=2, dim=1))
            logit_distance_loss_recon_advs = torch.mean(torch.norm(f_recon-f_adv, p=2, dim=1))
            logit_distance_loss_recon_cleans = torch.mean(torch.norm(f_recon-f_clean, p=2, dim=1))
            
            # distance_calculation(advs, cleans, cleans.shape[0], distance_dir, 'image/', 'adv_clean/', args, attack_name, model_name)
            # distance_calculation(advs, denoised, cleans.shape[0], distance_dir, 'image/','adv_denoised/', args, attack_name, model_name) 
            # distance_calculation(cleans, denoised, cleans.shape[0], distance_dir, 'image/' ,'clean_denoised/', args, attack_name, model_name) 
            # distance_calculation(f_adv, f_clean, cleans.shape[0], distance_dir, 'logit_clean/', 'logit_adv_clean/', args, attack_name, model_name)
            # distance_calculation(f_adv, f_denoised, cleans.shape[0], distance_dir, 'logit_clean/',  'logit_adv_denoised/', args, attack_name, model_name)
            # distance_calculation(f_clean, f_denoised, cleans.shape[0], distance_dir, 'logit_clean/', 'logit_clean_denoised/', args, attack_name, model_name)   
            # distance_calculation(f_adv, f_recon, cleans.shape[0], distance_dir, 'logit_recon/', 'logit_adv_recon/', args, attack_name, model_name)
            # distance_calculation(f_clean, f_recon, cleans.shape[0], distance_dir, 'logit_recon/', 'logit_clean_recon/', args, attack_name, model_name)
            # distance_calculation(f_adv, f_clean, cleans.shape[0], distance_dir, 'logit_recon/', 'logit_adv_clean/', args, attack_name, model_name)
            


            # # ----------------- angle calculation start ---------------------
            # cos = nn.CosineSimilarity(dim=1, eps=1e-6)

            # perturbation_view = perturbation.view(cleans.shape[0],-1)
            # recon_perturb = advs - denoised
            # recon_perturb_view = recon_perturb.view(cleans.shape[0],-1)
            # simi = cos(perturbation_view, recon_perturb_view)  # the angle at adv
            # angle_adv = torch.rad2deg(torch.acos(simi))
            # angle_adv_array = angle_adv.cpu().numpy()
            # print(angle_adv_array.shape)
            
            # np.save(angle_dir+ 'adv/'+ '{}_{}_{}.npy'.format(args.intensity, attack_name, model_name), angle_adv_array)
            # simi = cos(perturbation_view, (denoised-cleans).view(cleans.shape[0],-1))
            # angle_clean = torch.rad2deg(torch.acos(simi))
            # angle_clean_array = angle_clean.cpu().numpy()
            # np.save(angle_dir+ 'clean/'+ '{}_{}_{}.npy'.format(args.intensity, attack_name, model_name), angle_clean_array)

            # f_perturbation = f_adv - f_clean
            # f_perturbation = f_perturbation.view(cleans.shape[0],-1)
            # f_recon_perturb = f_adv - f_denoised
            # f_recon_perturb = f_recon_perturb.view(cleans.shape[0],-1)
            # simi = cos(f_perturbation, f_recon_perturb)  # the angle at f_adv
            # angle_logit_clean_denoised = torch.rad2deg(torch.acos(simi))
            # angle_logit_clean_denoised_array = angle_logit_clean_denoised.cpu().numpy()
            # np.save(angle_dir+ 'logit_clean_denoised/'+ '{}_{}_{}.npy'.format(args.intensity, attack_name, model_name), angle_logit_clean_denoised_array)
            # # print(angle_logit_adv_array)
            # simi = cos((f_denoised-f_clean).view(cleans.shape[0],-1), f_perturbation.view(cleans.shape[0],-1))
            # angle_logit_adv_denoised = torch.rad2deg(torch.acos(simi))
            # angle_logit_adv_denoised_array = angle_logit_adv_denoised.cpu().numpy()
            # np.save(angle_dir + 'logit_adv_denoised/'+ '{}_{}_{}.npy'.format(args.intensity, attack_name, model_name), angle_logit_adv_denoised_array)


            # simi = cos((f_recon - f_clean).view(cleans.shape[0],-1), (f_adv - f_clean).view(cleans.shape[0],-1))
            # angle_logit_adv_recon = torch.rad2deg(torch.acos(simi))
            # angle_logit_adv_recon_array = angle_logit_adv_recon.cpu().numpy()
            # np.save(angle_dir + 'logit_adv_recon/'+ '{}_{}_{}.npy'.format(args.intensity, attack_name, model_name), angle_logit_adv_recon_array)

            # simi = cos((f_adv - f_clean).view(cleans.shape[0], -1), (f_adv - f_recon).view(cleans.shape[0], -1))
            # angle_logit_clean_recon = torch.rad2deg(torch.acos(simi))
            # angle_logit_clean_recon_array = angle_logit_clean_recon.cpu().numpy()
            # np.save(angle_dir + 'logit_clean_recon/'+ '{}_{}_{}.npy'.format(args.intensity, attack_name, model_name), angle_logit_clean_recon_array)
            # # ----------------- angle calculation end ---------------------

            # measure accuracy and record loss

            distance_losses_denoised_cleans.update(distance_loss_denoised_cleans, cleans.size(0)) # D(x') x L2
            distance_losses_cleans_advs.update(distance_loss_cleans_advs, cleans.size(0))
            distance_losses_denoised_advs.update(distance_loss_denoised_advs, cleans.size(0))

            logit_distance_losses_denoised_advs.update(logit_distance_loss_denoised_advs, cleans.size(0))
            logit_distance_losses_cleans_advs.update(logit_distance_loss_cleans_advs, cleans.size(0))
            logit_distance_losses_denoised_cleans.update(logit_distance_loss_denoised_cleans, cleans.size(0))
            logit_distance_losses_recon_cleans.update(logit_distance_loss_recon_cleans, cleans.size(0))
            logit_distance_losses_recon_advs.update(logit_distance_loss_recon_advs, cleans.size(0))
            
    
            losses_denoised_advs.update(loss_denoised_advs.item(), cleans.size(0))
            losses_cleans_advs.update(loss_cleans_advs.item(), cleans.size(0))
            losses_denoised_cleans.update(loss_denoised_cleans.item(), cleans.size(0))
            losses_denoised_advs_L1.update(loss_denoised_advs_L1.item(), cleans.size(0))
            losses_cleans_advs_L1.update(loss_cleans_advs_L1.item(), cleans.size(0))
            losses_denoised_cleans_L1.update(loss_denoised_cleans_L1.item(), cleans.size(0))
            top1.update(acc_denoised.item(), cleans.size(0))
            top1_perturb.update(acc_perturb.item(), cleans.size(0))
            

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # if i % print_freq == 0:
            #     log = 'Test: [{0}/{1}]\t'' \
            #     ''Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'' \
            #     ''Data {data_time.val:.3f} ({data_time.avg:.3f})\t'' \
            #     ''Loss_denoised_advs {loss_denoised_advs.val:.4f} ({loss_denoised_advs.avg:.4f})\t'' \
            #     ''Loss_cleans_advs {loss_cleans_advs.val:.4f} ({loss_cleans_advs.avg:.4f})\t'' \
            #     ''Loss_denoised_cleans {loss_denoised_cleans.val:.4f} ({loss_denoised_cleans.avg:.4f})\t'' \
            #     ''Loss_denoised_advs_L1 {loss_denoised_advs_L1.val:.4f} ({loss_denoised_advs_L1.avg:.4f})\t'' \
            #     ''Loss_cleans_advs_L1 {loss_cleans_advs_L1.val:.4f} ({loss_cleans_advs_L1.avg:.4f})\t'' \
            #     ''Loss_denoised_cleans_L1 {loss_denoised_cleans_L1.val:.4f} ({loss_denoised_cleans_L1.avg:.4f})\t'' \
            #     ''Acc@denosied {top1.val:.3f} ({top1.avg:.3f})\t'' \
            #     ''Acc@perturbed {top1_perturb.val:.3f} ({top1_perturb.avg:.3f})\n'.format(
            #         i, len(loader), batch_time=batch_time,
            #         data_time=data_time, loss_denoised_advs=losses_denoised_advs, loss_cleans_advs = losses_cleans_advs, loss_denoised_cleans = losses_denoised_cleans, loss_denoised_advs_L1 =losses_denoised_advs_L1, loss_cleans_advs_L1 = losses_cleans_advs_L1, loss_denoised_cleans_L1 = losses_denoised_cleans_L1,top1=top1, top1_perturb=top1_perturb)

            #     print(log)

            #     out = open(outputFile, 'a')
            #     out.write(log)
            #     out.close()
        # return (losses_denoised_advs.avg, losses_cleans_advs.avg, losses_denoised_cleans.avg, losses_denoised_advs_L1.avg, losses_cleans_advs_L1.avg, losses_denoised_cleans_L1.avg, top1.avg, top1_perturb.avg, image_num, distance_losses_denoised_cleans.avg, distance_losses_cleans_advs.avg, distance_losses_denoised_advs.avg)
        return (top1.avg, top1_perturb.avg, image_num, distance_losses_denoised_cleans.avg, distance_losses_cleans_advs.avg, distance_losses_denoised_advs.avg, logit_distance_losses_denoised_cleans.avg, logit_distance_losses_cleans_advs.avg, logit_distance_losses_denoised_advs.avg, logit_distance_losses_recon_advs.avg, logit_distance_losses_recon_cleans.avg)

def imsave(img, index, name):
    # inv_normalize = transforms.Normalize(
    #     mean=[-0.4914/0.2023, -0.4822/0.1994, -0.4465/0.2010],
    #     std=[1/0.2023, 1/0.1994, 1/0.2010])
    # inv_tensor = inv_normalize(img)
    npimg = img.cpu().numpy()   # convert from tensor
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    if not os.path.isdir('example_visualize/img{}/'.format(index)):
        os.mkdir('example_visualize/img{}/'.format(index))
    plt.savefig('example_visualize/img{}/{}.png'.format(index,name))

def PSR(clean, adv, classifier, name):
    # print(torch.unsqueeze(clean,0).shape)
    t0 = classifier(torch.unsqueeze(clean,0)).argmax(1).detach().clone()
    t = classifier(torch.unsqueeze(adv,0)).argmax(1).detach().clone()
    _, length, width = clean.shape
    d0 = np.zeros([length,width])
    dt = np.zeros([length,width])
    Z_adv = classifier(torch.unsqueeze(adv,0))
    Z_adv_t0 = Z_adv[0][t0]
    # print(Z_adv.shape)
    Z_adv_t = Z_adv[0][t]
    for i in range(length):
        for j in range(width):
            ablation = adv
            ablation[:,i,j] = clean[:,i,j]
            Z_ablation = classifier(torch.unsqueeze(ablation,0))
            Z_ablation_t0 = Z_ablation[0][t0].detach().clone()
            Z_ablation_t = Z_ablation[0][t].detach().clone()
            # print(Z_ablation_t0-Z_adv_t0)
            d0[i][j] = np.max([Z_ablation_t0-Z_adv_t0,0.01])
            dt[i][j] = np.max([Z_adv_t-Z_ablation_t,0.01])
    PSR = np.log2(dt/d0)
    # S = dt+d0
    # S = S >= np.min(heapq.nlargest(int(len(S.flatten())*0.7),S.flatten()))
    PSR = PSR>=1
    
    # print(PSR)

    fig,ax=plt.subplots()
    
    npimg = clean.cpu().numpy()   # convert from tensor

    ax.imshow(np.transpose(npimg, (1, 2, 0)))
    ax.imshow(PSR,alpha=0.5,cmap="gray")
    plt.savefig('example_visualize/PSR_{}.png'.format(name))
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--dataset', type=str, choices=DATASETS, required=True)
    parser.add_argument('--denoiser', type=str, default='',
                        help='Path to a denoiser ', required=True)
    # parser.add_argument('--clf', type=str, default='',
    #                     help='Pretrained classificaiton model.', required=True)
    # parser.add_argument('--clf_gt', type=str, default='',
    #                     help='Pretrained groundtruth model.', required=True)
    parser.add_argument('--outdir', type=str, default='tmp_out/', help='folder to save model and training log)')
    parser.add_argument('--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--batch', default=32, type=int, metavar='N',
                        help='batchsize (default: 256)')
    parser.add_argument('--gpu', default=None, type=str,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--print-freq', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--noise_sd', default=0.0, type=float,
                        help="standard deviation of noise distribution for data augmentation")
    parser.add_argument('--test-subset', action='store_true',
                        help='evaluate only a predifined subset ~(1%) of the test set')
    parser.add_argument('--gamma1',type=float, default=0.01, help='the coefficient of reconstruction accuracy')
    parser.add_argument('--lambda1',type=float, default=0.01, help='the coefficient of adv reconstruction accuracy')
    parser.add_argument('--mu1', default = 0.1, type = float, help='l1 sparsity for reconstructed perturbation advs-denoised')
    parser.add_argument('--reg_data', default = 'all', type=str, help='The type of data for fine tuning, all, ori, robust')
    parser.add_argument('--attack_method', default = 'PGD,FGSM,CW', type = str, help = 'The attack attacks, PGD, FGSM, CW')
    parser.add_argument('--victim_model', default = 'resnet18', type = str, help ='The victim model, resnet18, resnet50, vgg16, vgg19, incptv3')
    # parser.add_argument('--intensity', default = 'medium', type = str, help ='The attack density, medium, strong, weak,out')
    parser.add_argument('--transform', default = 'rotation,flip', type = str, help ='The transformations for data augmentation')
    parser.add_argument('--test_DTx',default=0, type = int, help='whether test the performance of DTx or not')
    parser.add_argument('--test_TDx',default=0, type = int, help='whether test the performance of TDx or not')
    parser.add_argument('--intensity', default = 'weak,medium,strong', type = str, help ='The attack density, medium, strong, weak,out')
    parser.add_argument('--img_folder', default='', type=str,
                    help='path to the test data')
    parser.add_argument('--root', default='', type=str,
                    help='path to the root of the training code')
    parser.add_argument('--robust_res50_path', default='', type=str, help='The path of the checkpoint for the victim model robust res50')
    args = parser.parse_args()
    
    main(args)
