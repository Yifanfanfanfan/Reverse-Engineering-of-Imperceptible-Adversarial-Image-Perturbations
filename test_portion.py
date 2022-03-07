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
    if torch.cuda.is_available():
        denoiser = torch.nn.DataParallel(denoiser).cuda()
    denoiser.load_state_dict(checkpoint['state_dict'])
       
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
    
    root = ''
    attack_method = list(map(lambda x: str(x), args.attack_method.split(",")))
    victim_model = list(map(lambda x: str(x), args.victim_model.split(",")))
    intensity = list(map(lambda x: str(x), args.intensity.split(",")))

    clf_losses_denoised_advs = np.zeros((len(attack_method), len(victim_model)))  #D(x') x' L2
    clf_losses_cleans_advs =  np.zeros((len(attack_method), len(victim_model)))  # x' x  L2
    clf_losses_denoised_cleans = np.zeros((len(attack_method), len(victim_model))) # D(x') x L2
    clf_losses_denoised_advs_L1 = np.zeros((len(attack_method), len(victim_model)))  #D(x') x' L1
    clf_losses_cleans_advs_L1 = np.zeros((len(attack_method), len(victim_model)))  # x' x  L1
    clf_losses_denoised_cleans_L1 = np.zeros((len(attack_method), len(victim_model))) # D(x') x  L1
    
    clf_acc = np.zeros((len(attack_method), len(victim_model)))
    clf_acc_perturb = np.zeros((len(attack_method), len(victim_model)))


    # losses_adv_a_advs = np.zeros(args.fig_nodes_num)
    # losses_adv_a_cleans = np.zeros(args.fig_nodes_num)
    # losses_adv_a_advs_L1 = np.zeros(args.fig_nodes_num)
    # losses_adv_a_cleans_L1 = np.zeros(args.fig_nodes_num)
    # accs_adv_a = np.zeros(args.fig_nodes_num)
    # suc_rates_adv_a = np.zeros(args.fig_nodes_num)
    # logits_losses_denoised_clean = np.zeros(args.fig_nodes_num)
    # logits_losses_denoised_adv = np.zeros(args.fig_nodes_num)
    # logits_losses_clean_adv = np.zeros(args.fig_nodes_num)
    

    # logits_losses_recon_adv_clean = np.zeros(args.fig_nodes_num)
    # logits_losses_recon_adv_adv = np.zeros(args.fig_nodes_num)
    # logits_losses_adv_c_clean = np.zeros(args.fig_nodes_num)
    # logits_losses_adv_c_adv = np.zeros(args.fig_nodes_num)

    accs_adv_c = np.zeros((args.fig_nodes_num,len(victim_model)))
    suc_rates_adv_c = np.zeros((args.fig_nodes_num, len(victim_model)))
    accs_recon_adv = np.zeros((args.fig_nodes_num, len(victim_model)))
    suc_rates_recon_adv = np.zeros((args.fig_nodes_num, len(victim_model)))
    distance_losses_cleans_recon_adv = np.zeros((args.fig_nodes_num, len(victim_model)))
    distance_losses_cleans_advs_c = np.zeros((args.fig_nodes_num, len(victim_model)))
    logit_distances_cleans_recon_adv = np.zeros((args.fig_nodes_num, len(victim_model)))
    logit_distances_cleans_advs_c = np.zeros((args.fig_nodes_num,len(victim_model)))
    

    
    
    model_num = 0
    for model_name in victim_model: #res18,res50,vgg16,vgg19,incptv3
        print(model_name)
        test_data_list = list()
        for  attack_name in attack_method: #PGD,FGSM,CW
            for intensity_name in intensity:
                text_test_list = os.path.join(root, 'test_list_portion_{}_{}_{}.txt'.format(model_name, attack_name, intensity_name))
                img_folder = '/home/yifan/github/RED_Denoising/RED_Denoising/code/4typesdenoisedtestdata/4typesdenoisedtestdata/'
                # img_test_clean = '/home/yifan/github/RED_Denoising/RED_Denoising/code/denoise/' + model_name + '/testclean/' + args.intensity
                # img_test_adv = '/home/yifan/github/RED_Denoising/RED_Denoising/code/denoise/' + model_name + '/test/' + attack_name + '/' + args.intensity  
                img_test_clean = img_folder + attack_name + model_name + '/test4types/clean'
                img_test_adv = img_folder + attack_name + model_name + '/test4types/' + intensity_name
                tt.gen_txt_new_new('test_list_portion_{}_{}_{}.txt'.format(model_name, attack_name, intensity_name), img_test_clean, img_test_adv)
                test_data_method_model = RD.FaceDataset(text_test_list)


                test_data_list.append(RD.FaceDataset(text_test_list))
                multiple_dataset = data.ConcatDataset(test_data_list)
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
        print(test_data_list)

        classification_criterion = CrossEntropyLoss(size_average=None, reduce=None, reduction = 'mean').cuda()
        test_loader = DataLoader(dataset=multiple_dataset, batch_size=args.batch, num_workers=args.workers, pin_memory=False)

            
            
        # ------------pertubation ratio begin---------------
        for a in range(args.fig_nodes_num):
            
            accs_recon_adv[a,model_num], accs_adv_c[a,model_num], suc_rates_recon_adv[a,model_num], suc_rates_adv_c[a, model_num],\
            distance_losses_cleans_recon_adv[a,model_num], distance_losses_cleans_advs_c[a,model_num],\
            logit_distances_cleans_recon_adv[a,model_num], logit_distances_cleans_advs_c[a,model_num]\
            = test_with_classifier_c(test_loader, denoiser, classification_criterion, args.noise_sd, args.print_freq, clf,clf_gt, a)
            print('The average MSE between recon_adv images and clean images for a{} model {} is {}'.format(a,model_name, distance_losses_cleans_recon_adv[a,model_num]))
            print('The average MSE between adv_c images and clean images for a{} is {}'.format(a,distance_losses_cleans_advs_c[a,model_num]))
            print('The average MSE between f_recon_adv and f_clean images for a{} is {}'.format(a,logit_distances_cleans_recon_adv[a,model_num]))
            print('The average MSE between f_adv_c and f_clean images for a{} is {}'.format(a,logit_distances_cleans_advs_c[a,model_num]))
            print('The accuracy of adv_c for a{} is {}'.format(a, accs_adv_c[a,model_num]))
            print('The accuracy of recon_adv for a{} is {}'.format(a, accs_recon_adv[a,model_num]))
            print('The attack success rate of adv_c for a{} is {}'.format(a, suc_rates_adv_c[a,model_num]))
            print('The attack success rate of recon_adv for a{} is {}'.format(a, suc_rates_recon_adv[a,model_num]))
        # ------------pertubation ratio end---------------
        model_num = model_num + 1
    

    # if not os.path.isdir('perturbation_interpolation_test/'):
    #     os.mkdir('perturbation_interpolation_test/')
    # np.savez("perturbation_interpolation_test/MSE_re_denoise_RED_{}.npz".format(args.fig_nodes_num),distance_losses_cleans_recon_adv, distance_losses_cleans_advs_c)
    # # np.savez("perturbation_interpolation_test/MAE_re_denoise_RED_{}.npz".format(args.fig_nodes_num),losses_adv_a_advs_L1, losses_adv_a_cleans_L1)
    # np.savez("perturbation_interpolation_test/acc_re_denoise_RED_{}.npz".format(args.fig_nodes_num), accs_recon_adv, accs_adv_c)
    # np.savez("perturbation_interpolation_test/suc_rate_re_denoise_RED_{}.npz".format(args.fig_nodes_num), suc_rates_recon_adv, suc_rates_adv_c)
    # np.savez("perturbation_interpolation_test/logits_loss_re_denoise_RED_{}.npz".format(args.fig_nodes_num),logit_distances_cleans_recon_adv, logit_distances_cleans_advs_c)


    return None
    

    



def test_with_classifier(loader: DataLoader, denoiser: torch.nn.Module, criterion, noise_sd: float, print_freq: int, classifier: torch.nn.Module, classifier_gt: torch.nn.Module):
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

    

    end = time.time()

    # switch to eval mode
    classifier.eval()
    classifier_gt.eval()
    if denoiser:
        denoiser.eval()

    with torch.no_grad():
        
        for i, (cleans, advs) in enumerate(loader):
        # measure data loading time
            data_time.update(time.time() - end)
            perturbation = cleans - advs
            
            cleans = cleans.cuda().to(dtype=torch.float)
            advs = advs.cuda().to(dtype=torch.float) 
                
            
            if denoiser is not None:
                denoised = denoiser(advs)
            
            adv_outputs = classifier(advs - denoised + cleans)
            f_denoised = classifier(denoised)
            clean_label = classifier(cleans)
            adv_label = classifier(advs)

            adv_outputs_label = adv_outputs.argmax(1).detach().clone()
            
            f_denoised_label = f_denoised.argmax(1).detach().clone()

            clean_label = clean_label.argmax(1).detach().clone()

            adv_label = adv_label.argmax(1).detach().clone()
            
            
            loss_denoised_advs_L1 = L1(denoised, advs)
            loss_cleans_advs_L1 = L1(cleans, advs)
            loss_denoised_cleans_L1 = L1(denoised, cleans)
            loss_denoised_advs = MSE(denoised, advs)
            loss_cleans_advs = MSE(cleans, advs)
            loss_denoised_cleans = MSE(denoised, cleans)
            # acc_denoised = sum(clean_label_byclf==clean_label)/cleans.size(0)
            acc_denoised = sum(f_denoised_label==clean_label)/cleans.size(0)
            acc_perturb = sum(adv_outputs_label==adv_label)/cleans.size(0)
            # print(acc_denoised)
            # print(acc_perturb)


            # measure accuracy and record loss
            # acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            # losses.update(loss.item(), cleans.size(0))
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

            if i % print_freq == 0:
                log = 'Test: [{0}/{1}]\t'' \
                ''Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'' \
                ''Data {data_time.val:.3f} ({data_time.avg:.3f})\t'' \
                ''Loss_denoised_advs {loss_denoised_advs.val:.4f} ({loss_denoised_advs.avg:.4f})\t'' \
                ''Loss_cleans_advs {loss_cleans_advs.val:.4f} ({loss_cleans_advs.avg:.4f})\t'' \
                ''Loss_denoised_cleans {loss_denoised_cleans.val:.4f} ({loss_denoised_cleans.avg:.4f})\t'' \
                ''Loss_denoised_advs_L1 {loss_denoised_advs_L1.val:.4f} ({loss_denoised_advs_L1.avg:.4f})\t'' \
                ''Loss_cleans_advs_L1 {loss_cleans_advs_L1.val:.4f} ({loss_cleans_advs_L1.avg:.4f})\t'' \
                ''Loss_denoised_cleans_L1 {loss_denoised_cleans_L1.val:.4f} ({loss_denoised_cleans_L1.avg:.4f})\t'' \
                ''Acc@denosied {top1.val:.3f} ({top1.avg:.3f})\t'' \
                ''Acc@perturbed {top1_perturb.val:.3f} ({top1_perturb.avg:.3f})\n'.format(
                    i, len(loader), batch_time=batch_time,
                    data_time=data_time, loss_denoised_advs=losses_denoised_advs, loss_cleans_advs = losses_cleans_advs, loss_denoised_cleans = losses_denoised_cleans, loss_denoised_advs_L1 =losses_denoised_advs_L1, loss_cleans_advs_L1 = losses_cleans_advs_L1, loss_denoised_cleans_L1 = losses_denoised_cleans_L1,top1=top1, top1_perturb=top1_perturb)

                print(log)

                out = open(outputFile, 'a')
                out.write(log)
                out.close()

        return (losses_denoised_advs.avg, losses_cleans_advs.avg, losses_denoised_cleans.avg, losses_denoised_advs_L1.avg, losses_cleans_advs_L1.avg, losses_denoised_cleans_L1.avg, top1.avg, top1_perturb.avg)
        


# test with true perturbation
def test_with_classifier_a(loader: DataLoader, denoiser: torch.nn.Module, criterion, noise_sd: float, print_freq: int, classifier: torch.nn.Module, classifier_gt: torch.nn.Module, a):
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

    losses_adv_a_advs = AverageMeter()
    losses_adv_a_cleans = AverageMeter()
    losses_adv_a_advs_L1 = AverageMeter()
    losses_adv_a_cleans_L1 = AverageMeter()
    accs_adv_a = AverageMeter()
    suc_rates_adv_a = AverageMeter()

    end = time.time()

    # switch to eval mode
    classifier.eval()
    classifier_gt.eval()
    if denoiser:
        denoiser.eval()

    with torch.no_grad():
        for i, (cleans, advs) in enumerate(loader):
        # measure data loading time
            data_time.update(time.time() - end)
            perturbation = advs - cleans
            adv_a = cleans + a/20 * perturbation
            cleans = cleans.cuda().to(dtype=torch.float)
            advs = advs.cuda().to(dtype=torch.float) 
            adv_a = adv_a.cuda().to(dtype=torch.float)
            clean_label = classifier(cleans)
            adv_label = classifier(advs)
            adv_a_label = classifier(adv_a)
            clean_label = clean_label.argmax(1).detach().clone()
            adv_label = adv_label.argmax(1).detach().clone()
            adv_a_label = adv_a_label.argmax(1).detach().clone()
            acc_adv_a = sum(adv_a_label==clean_label)/cleans.size(0)
            suc_rate_adv_a = sum(adv_a_label==adv_label)/cleans.size(0)
            loss_adv_a_advs_L1 = L1(adv_a, advs)
            loss_adv_a_cleans_L1 = L1(adv_a, cleans)
            loss_adv_a_advs = MSE(adv_a, advs)
            loss_adv_a_cleans = MSE(adv_a, cleans)

            losses_adv_a_advs.update(loss_adv_a_advs.item(), cleans.size(0))
            losses_adv_a_cleans.update(loss_adv_a_cleans.item(), cleans.size(0))
            losses_adv_a_advs_L1.update(loss_adv_a_advs_L1.item(), cleans.size(0))
            losses_adv_a_cleans_L1.update(loss_adv_a_cleans_L1.item(), cleans.size(0))
            accs_adv_a.update(acc_adv_a.item(), cleans.size(0))
            suc_rates_adv_a.update(suc_rate_adv_a.item(), cleans.size(0))

    return (losses_adv_a_advs.avg, losses_adv_a_cleans.avg, losses_adv_a_advs_L1.avg, losses_adv_a_cleans_L1.avg, accs_adv_a.avg, suc_rates_adv_a.avg)

# test with reconstructed perturbation
def test_with_classifier_b(loader: DataLoader, denoiser: torch.nn.Module, criterion, noise_sd: float, print_freq: int, classifier: torch.nn.Module, classifier_gt: torch.nn.Module, a):
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

    losses_adv_a_advs = AverageMeter()
    losses_adv_a_cleans = AverageMeter()
    losses_adv_a_advs_L1 = AverageMeter()
    losses_adv_a_cleans_L1 = AverageMeter()
    accs_adv_a = AverageMeter()
    suc_rates_adv_a = AverageMeter()

    end = time.time()

    # switch to eval mode
    classifier.eval()
    classifier_gt.eval()
    if denoiser:
        denoiser.eval()

    with torch.no_grad():
        for i, (cleans, advs) in enumerate(loader):
        # measure data loading time
            data_time.update(time.time() - end)
            cleans = cleans.cuda().to(dtype=torch.float)
            advs = advs.cuda().to(dtype=torch.float) 
            if denoiser is not None:
                denoised = denoiser(advs)
            re_perturbation = advs - denoised
            adv_a = cleans + a/20 * re_perturbation
            clean_label = classifier(cleans)
            adv_label = classifier(advs)
            adv_a_label = classifier(adv_a)
            clean_label = clean_label.argmax(1).detach().clone()
            adv_label = adv_label.argmax(1).detach().clone()
            adv_a_label = adv_a_label.argmax(1).detach().clone()
            acc_adv_a = sum(adv_a_label==clean_label)/cleans.size(0)
            suc_rate_adv_a = sum(adv_a_label==adv_label)/cleans.size(0)
            loss_adv_a_advs_L1 = L1(adv_a, advs)
            loss_adv_a_cleans_L1 = L1(adv_a, cleans)
            loss_adv_a_advs = MSE(adv_a, advs)
            loss_adv_a_cleans = MSE(adv_a, cleans)

            losses_adv_a_advs.update(loss_adv_a_advs.item(), cleans.size(0))
            losses_adv_a_cleans.update(loss_adv_a_cleans.item(), cleans.size(0))
            losses_adv_a_advs_L1.update(loss_adv_a_advs_L1.item(), cleans.size(0))
            losses_adv_a_cleans_L1.update(loss_adv_a_cleans_L1.item(), cleans.size(0))
            accs_adv_a.update(acc_adv_a.item(), cleans.size(0))
            suc_rates_adv_a.update(suc_rate_adv_a.item(), cleans.size(0))

    return (losses_adv_a_advs.avg, losses_adv_a_cleans.avg, losses_adv_a_advs_L1.avg, losses_adv_a_cleans_L1.avg, accs_adv_a.avg, suc_rates_adv_a.avg)

# denoise with part of perturbation
def test_with_classifier_c(loader: DataLoader, denoiser: torch.nn.Module, criterion, noise_sd: float, print_freq: int, classifier: torch.nn.Module, classifier_gt: torch.nn.Module, a):
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

    accs_adv_c = AverageMeter()
    suc_rates_adv_c = AverageMeter()
    accs_recon_adv = AverageMeter()
    suc_rates_recon_adv = AverageMeter()


    distance_losses_cleans_recon_adv = AverageMeter()
    distance_losses_cleans_advs_c = AverageMeter()
    
    logit_distances_cleans_recon_adv = AverageMeter()
    logit_distances_cleans_advs_c = AverageMeter()

    end = time.time()

    # switch to eval mode
    classifier.eval()
    classifier_gt.eval()
    if denoiser:
        denoiser.eval()

    with torch.no_grad():
        for i, (cleans, advs) in enumerate(loader):
        # measure data loading time
            data_time.update(time.time() - end)
            cleans = cleans.cuda().to(dtype=torch.float)
            advs = advs.cuda().to(dtype=torch.float)

            perturbation = advs - cleans
            print(a/(args.fig_nodes_num-1))
            advs_c = cleans + a/(args.fig_nodes_num-1) * perturbation  #parially introplated adv images

            if denoiser is not None:
                denoised = denoiser(advs_c)
            
            # recon_adv = advs - denoised + cleans
            recon_adv = advs_c - denoised + cleans

            f_clean = classifier(cleans)
            f_adv = classifier(advs)
            f_denoised = classifier(denoised)
            f_recon_adv = classifier(recon_adv)
            f_adv_c = classifier(advs_c)

            F_clean = f_clean.argmax(1).detach().clone()
            F_adv = f_adv.argmax(1).detach().clone()
            F_denoised = f_denoised.argmax(1).detach().clone()
            F_recon_adv = f_recon_adv.argmax(1).detach().clone()
            F_adv_c = f_adv_c.argmax(1).detach().clone()
            # acc_adv_a = sum(denoised_label==clean_label)/cleans.size(0)
            acc_recon_adv= sum(F_recon_adv==F_clean)/cleans.size(0)
            suc_rate_recon_adv = sum(F_recon_adv==F_adv)/cleans.size(0)

            acc_adv_c = sum(F_adv_c == F_clean)/cleans.size(0)
            suc_rate_adv_c = sum(F_adv_c == F_adv)/cleans.size(0)


            # loss_adv_a_advs_L1 = L1(recon_adv, advs_c)
            # loss_adv_a_cleans_L1 = L1(recon_adv, cleans)
            logits_loss_denoised_clean = MSE(f_denoised, f_clean)
            logits_loss_denoised_adv = MSE(f_denoised, f_adv)
            logits_loss_clean_adv = MSE(f_clean, f_adv)
            logits_loss_recon_adv_clean = MSE(f_recon_adv, f_clean)
            logits_loss_recon_adv_adv = MSE(f_recon_adv, f_adv)


            # distance_loss_denoised_cleans = torch.zeros(cleans.shape[0]).cuda().to(dtype=torch.float)
            # distance_loss_denoised_advs = torch.zeros(cleans.shape[0]).cuda().to(dtype=torch.float)
            distance_loss_cleans_recon_adv = torch.zeros(cleans.shape[0]).cuda().to(dtype=torch.float)
            distance_loss_cleans_advs_c = torch.zeros(cleans.shape[0]).cuda().to(dtype=torch.float)
            logit_distance_cleans_recon_adv = torch.zeros(cleans.shape[0]).cuda().to(dtype=torch.float)
            logit_distance_cleans_advs_c = torch.zeros(cleans.shape[0]).cuda().to(dtype=torch.float)

            # distance_loss_adv_c_advs = torch.zeros(cleans.shape[0]).cuda().to(dtype=torch.float)
            # distance_loss_recon_adv_advs = torch.zeros(cleans.shape[0]).cuda().to(dtype=torch.float)

            # logit_distance_recon_adv_advs = torch.zeros(cleans.shape[0]).cuda().to(dtype=torch.float)
            # logit_distance_advs_c_advs = torch.zeros(cleans.shape[0]).cuda().to(dtype=torch.float)
            
            MSE_dis = MSELoss(size_average=None, reduce=None, reduction='sum').cuda()
            for num in range(cleans.shape[0]):
                distance_loss_cleans_recon_adv[num] = MSE_dis(cleans[num], recon_adv[num])
                distance_loss_cleans_advs_c[num] = MSE_dis(cleans[num], advs_c[num])
                logit_distance_cleans_recon_adv[num] = MSE_dis(f_clean[num], f_recon_adv[num])
                logit_distance_cleans_advs_c[num] = MSE_dis(f_clean[num], f_adv_c[num])

            distance_loss_cleans_recon_adv = torch.sqrt(distance_loss_cleans_recon_adv)
            distance_loss_cleans_advs_c = torch.sqrt(distance_loss_cleans_advs_c)
    
            logit_distance_cleans_recon_adv= torch.sqrt(logit_distance_cleans_recon_adv)
            logit_distance_cleans_advs_c = torch.sqrt(logit_distance_cleans_advs_c)

            # distance_loss_denoised_cleans = torch.sqrt(distance_loss_denoised_cleans)
            # distance_loss_cleans_advs = torch.sqrt(distance_loss_cleans_advs)
            # distance_loss_denoised_advs = torch.sqrt(distance_loss_denoised_advs)

            # -------------groundtruth---------------
            # logits_loss_adv_c_adv = MSE(f_adv_c, f_adv)
            # logits_loss_adv_c_clean = MSE(f_adv_c, f_clean)

            # loss_adv_a_advs = MSE(recon_adv, advs_c)
            # loss_adv_a_cleans = MSE(recon_adv, cleans)


            # losses_adv_a_advs.update(loss_adv_a_advs.item(), cleans.size(0))
            # losses_adv_a_cleans.update(loss_adv_a_cleans.item(), cleans.size(0))
            # losses_adv_a_advs_L1.update(loss_adv_a_advs_L1.item(), cleans.size(0))
            # losses_adv_a_cleans_L1.update(loss_adv_a_cleans_L1.item(), cleans.size(0))
            accs_adv_c.update(acc_adv_c.item(), cleans.size(0))
            suc_rates_adv_c.update(suc_rate_adv_c.item(), cleans.size(0))
            accs_recon_adv.update(acc_recon_adv.item(), cleans.size(0))
            suc_rates_recon_adv.update(suc_rate_recon_adv.item(), cleans.size(0))


            distance_losses_cleans_recon_adv.update(torch.mean(distance_loss_cleans_recon_adv))
            distance_losses_cleans_advs_c.update(torch.mean(distance_loss_cleans_advs_c))
    
            logit_distances_cleans_recon_adv.update(torch.mean(logit_distance_cleans_recon_adv))
            logit_distances_cleans_advs_c.update(torch.mean(logit_distance_cleans_advs_c))

            
            # logits_losses_denoised_clean.update(logits_loss_denoised_clean.item(), cleans.size(0))
            # logits_losses_denoised_adv.update(logits_loss_denoised_adv.item(), cleans.size(0))
            # logits_losses_clean_adv.update(logits_loss_clean_adv.item(), cleans.size(0))
            # logits_losses_recon_adv_clean.update(logits_loss_recon_adv_clean.item(), cleans.size(0))
            # logits_losses_recon_adv_adv.update(logits_loss_recon_adv_adv.item(), cleans.size(0))
            # logits_losses_adv_c_clean.update(logits_loss_adv_c_clean.item(), cleans.size(0))
            # logits_losses_adv_c_adv.update(logits_loss_adv_c_adv.item(), cleans.size(0))
            

    return (accs_recon_adv.avg, accs_adv_c.avg, suc_rates_recon_adv.avg, suc_rates_adv_c.avg, distance_losses_cleans_recon_adv.avg, distance_losses_cleans_advs_c.avg, logit_distances_cleans_recon_adv.avg, logit_distances_cleans_advs_c.avg)


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
    parser.add_argument('--attack_method', default = 'PGD,FGSM,CW', type = str, help = 'The attack attacks, PGD, FGSM, CW')
    parser.add_argument('--victim_model', default = 'res18', type = str, help ='The victim model, res18, res50, vgg16, vgg19, incptv3')
    parser.add_argument('--fig_nodes_num', default = 6, type = int, help ='The number of nodes/portion partitions')
    parser.add_argument('--intensity', default = 'weak,medium,strong', type = str, help ='The attack density, medium, strong, weak,out')
    args = parser.parse_args()
    
    main(args)
