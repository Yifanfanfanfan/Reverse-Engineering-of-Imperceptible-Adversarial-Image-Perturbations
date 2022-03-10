# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# File for training denoisers with at most one classifier attached to

from torch.optim import lr_scheduler
from architectures import DENOISERS_ARCHITECTURES, get_architecture, IMAGENET_CLASSIFIERS
from datasets import get_dataset, DATASETS
from test_denoiser_aug import test_with_classifier
from torch.nn import MSELoss, CrossEntropyLoss, KLDivLoss, L1Loss
from torch.optim import SGD, Optimizer, Adam
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage
from train_utils import AverageMeter, accuracy, init_logfile, log, copy_code, requires_grad_
import torch.utils.data as data
import torch.nn as nn
from torchvision import models
import argparse
import datetime
import numpy as np
import os
import time
import torch
import torchvision
import dataset_txt_generate as tt
import RED_Dataset as RD
import random
from torchvision import transforms
import torchvision.transforms.functional as TF
from torch.utils.tensorboard import SummaryWriter  
from collections import OrderedDict

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--dataset', type=str, choices=DATASETS)
parser.add_argument('--arch', type=str, choices=DENOISERS_ARCHITECTURES)
parser.add_argument('--outdir', default='logs', type=str, help='folder to save denoiser and training log)')
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch', default=256, type=int, metavar='N',
                    help='batchsize (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=1e-6, type=float,
                    help='initial learning rate', dest='lr')
parser.add_argument('--lr_step_size', type=int, default=30,
                    help='How often to decrease learning by gamma.')
parser.add_argument('--gamma', type=float, default=0.1,
                    help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--gpu', default=None, type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--noise_sd', default=0.0, type=float,
                    help="standard deviation of noise distribution for data augmentation")
parser.add_argument('--objective', default='denoising', type=str,
                    help="the objective that is used to train the denoiser",
                    choices=['denoising', 'classification', 'stability'])
# parser.add_argument('--classifier', default='', type=str,
#                     help='path to the classifier used with the `classificaiton`'
#                          'or `stability` objectives of the denoiser.')
parser.add_argument('--surrogate_model', default='res18',type=str,help='the name of the surrogate model for regularization term calculation')
parser.add_argument('--robust_res50_path', default='', type=str, help='The path of the checkpoint for the victim model robust res50')
parser.add_argument('--pretrained-denoiser', default='', type=str,
                    help='path to a pretrained denoiser')
parser.add_argument('--advdata_dir', default='', type=str,
                    help='path to the training dataset')
parser.add_argument('--root', default='', type=str,
                    help='path to the root of the training code')
parser.add_argument('--optimizer', default='Adam', type=str,
                    help='SGD, Adam, or Adam then SGD', choices=['SGD', 'Adam', 'AdamThenSGD'])
parser.add_argument('--start-sgd-epoch', default=50, type=int,
                    help='[Relevent only to AdamThenSGD.] Epoch at which adam switches to SGD')
parser.add_argument('--start-sgd-lr', default=1e-5, type=float,
                    help='[Relevent only to AdamThenSGD.] LR at which SGD starts after Adam')
parser.add_argument('--resume', action='store_true',
                    help='if true, tries to resume training from an existing checkpoint')
parser.add_argument('--azure_datastore_path', type=str, default='',
                    help='Path to imagenet on azure')
parser.add_argument('--philly_imagenet_path', type=str, default='',
                    help='Path to imagenet on philly')
parser.add_argument('--gamma1',type=float, default=0.01, help='the coefficient of reconstruction accuracy')
parser.add_argument('--lambda1',type=float, default=0.01, help='the coefficient of adv reconstruction accuracy')
parser.add_argument('--finetune_stability', action='store_true', help='if true, tries to finetune with stability objective')
parser.add_argument('--l1orNOl1', default = 'NOl1', type = str, help='if l1, l1 regularizer for objective function')
parser.add_argument('--MAEorMSE', default = 'MAE', type = str, help='Choose MSE or MAE or Stability (neither of the former two) for loss function')
parser.add_argument('--mu1', default = 0.1, type = float, help='l1 sparsity for reconstructed perturbation advs-denoised')
parser.add_argument('--eta', default = 0, type = float, help='coefficient for the weight of the cosine angle')
parser.add_argument('--attack_method', default = 'PGD,FGSM,CW', type = str, help = 'The attack attacks, PGD, FGSM, CW')
parser.add_argument('--victim_model', default = 'res18', type = str, help ='The victim model, res18, res50, vgg16, vgg19, incptv3')
parser.add_argument('--data_portion', default = 0.1, type = float, help ='subset portion')
parser.add_argument('--aug', default='', type = str, help = 'augmentation types')
parser.add_argument('--aug2', default='', type = str, help = 'augmentation type 2s')
parser.add_argument('--reg_data', default = 'all', type=str, help='The type of data for fine tuning, all, ori, robust')
parser.add_argument("--multigpu", default=None, type=lambda x: [int(a) for a in x.split(",")], help="Which GPUs to use for multigpu training")
parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
args = parser.parse_args()

if args.azure_datastore_path:
    os.environ['IMAGENET_DIR_AZURE'] = os.path.join(args.azure_datastore_path, 'datasets/imagenet_zipped')
if args.philly_imagenet_path:
    os.environ['IMAGENET_DIR_PHILLY'] = os.path.join(args.philly_imagenet_path, './')

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
random.seed(0)

toPilImage = ToPILImage()


def set_gpu(args, model):
    assert torch.cuda.is_available(), "CPU-only experiments currently unsupported"
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    elif args.multigpu is None:
        device = torch.device("cpu")
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        print(f"=> Parallelizing on {args.multigpu} gpus")
        torch.cuda.set_device(args.multigpu[0])
        args.gpu = args.multigpu[0]
        model = torch.nn.DataParallel(model, device_ids=args.multigpu).cuda()
    if args.seed is None:
        cudnn.benchmark = True
    return model

def assign_learning_rate(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr

def cosine_lr(optimizer, args, **kwargs):
    def _lr_adjuster(epoch, iteration):
        # if epoch < args.warmup_length:
        #     lr = warmup_lr(args.lr, args.warmup_length, epoch)
        # else:
        e = epoch
        es = args.epochs
        lr = 0.5 * (1 + np.cos(np.pi * e / es)) * args.lr

        assign_learning_rate(optimizer, lr)
        
        return lr

    return _lr_adjuster

def main():
    
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    # Copy code to output directory
    copy_code(args.outdir)

    root = args.root
    best_loss = 10000.0


    # -------------------------------------------- step 1/5 : Data Loading -------------------------------------------

    print('-------------------------------------\nLoading Data...\n-------------------------------------\n')

    attack_method = list(map(lambda x: str(x), args.attack_method.split(",")))
    victim_model = list(map(lambda x: str(x), args.victim_model.split(",")))

    attack_method_victim_model_list = list()
    for model_name in victim_model:
        attack_method_victim_model_list.extend(list(map(lambda x: x+model_name, attack_method)))

    train_data_list = list()
    for item in attack_method_victim_model_list:
        txt_train_list = os.path.join(root, 'train_list.txt')
        advdata_dir = args.advdata_dir
        img_train_clean =  advdata_dir +  item + '/trainclean'
        img_train_adv =  advdata_dir + item + '/train'
        tt.gen_txt(txt_train_list, img_train_clean, img_train_adv)
        train_data_list.append(RD.FaceDataset(txt_train_list))

    multiple_dataset = data.ConcatDataset(train_data_list)


    
    example_num = len(multiple_dataset)
    idx_input = random.sample(range(0,example_num),int(example_num * args.data_portion))
    sub_dataset = data.Subset(multiple_dataset, idx_input)
    train_loader = DataLoader(dataset=sub_dataset, batch_size=args.batch, shuffle=True, num_workers=args.workers, pin_memory=False)



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
    # ------------------------------------ step 2/5 : Network Definition ------------------------------------
    if args.pretrained_denoiser:
        checkpoint = torch.load(args.pretrained_denoiser)
        assert checkpoint['arch'] == args.arch
        denoiser = get_architecture(checkpoint['arch'], args.dataset)
        denoiser = set_gpu(args, denoiser)
        denoiser.cuda()
        denoiser.load_state_dict(checkpoint['state_dict'])
    elif args.finetune_stability:
        checkpoint = torch.load('logs/gamma0lambda0checkpoint.pth.tar')
        assert checkpoint['arch'] == args.arch
        denoiser = get_architecture(checkpoint['arch'], args.dataset)
        denoiser.load_state_dict(checkpoint['state_dict'])
    else: # training a new denoiser
        denoiser = get_architecture(args.arch, args.dataset) 
    # denoiser = set_gpu(args, denoiser)
    # denoiser.cuda()
    if args.optimizer == 'Adam':
        optimizer = Adam(denoiser.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'SGD':
        optimizer = SGD(denoiser.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == 'AdamThenSGD':
        optimizer = Adam(denoiser.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=args.lr_step_size, gamma=args.gamma)

    starting_epoch = 0
    logfilename = os.path.join(args.outdir, 'log.txt')

    ## Resume from checkpoint if exists and if resume flag is True
    denoiser_path = os.path.join(args.outdir, 'checkpoint.pth.tar')
    if args.resume and os.path.isfile(denoiser_path):
        print("=> loading checkpoint '{}'".format(denoiser_path))
        checkpoint = torch.load(denoiser_path,
                                map_location=lambda storage, loc: storage)
        assert checkpoint['arch'] == args.arch
        starting_epoch = checkpoint['epoch']
        denoiser.load_state_dict(checkpoint['state_dict'])
        if starting_epoch >= args.start_sgd_epoch and args.optimizer == 'AdamThenSGD ':  # Do adam for few steps thaen continue SGD
            print("-->[Switching from Adam to SGD.]")
            args.lr = args.start_sgd_lr
            optimizer = SGD(denoiser.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
            scheduler = StepLR(optimizer, step_size=args.lr_step_size, gamma=args.gamma)

        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(denoiser_path, checkpoint['epoch']))
    else:
        if args.resume: print("=> no checkpoint found at '{}'".format(args.outdir))
        init_logfile(logfilename, "epoch\ttime\tlr\ttrainloss\ttestloss\ttestAcc")

    if args.objective == 'error':
        criterion = MSELoss(size_average=None, reduce=None, reduction='mean').cuda()
        best_loss = 1e6

    elif args.objective in ['classification', 'stability', 'denoising']:
        if args.surrogate_model == "incptv3":
            print('--load pretrained inceptv3--')
            clf = nn.Sequential(norm_layer,models.inception_v3(pretrained=True)).cuda().eval()
        elif args.surrogate_model == "vgg16":
            print('--load pretrained vgg16--')
            clf = nn.Sequential(norm_layer, models.vgg16(pretrained=True)).cuda().eval()
        elif args.surrogate_model == "vgg19":
            print('--load pretrained vgg19--')
            clf = nn.Sequential(norm_layer, models.vgg19(pretrained=True)).cuda().eval()
        elif args.surrogate_model == "res18":
            print('--load pretrained res18--')
            clf = nn.Sequential(norm_layer,models.resnet18(pretrained=True)).cuda().eval()
        elif args.surrogate_model == "res50":
            print('--load pretrained res50--')
            clf = nn.Sequential(norm_layer,models.resnet50(pretrained=True)).cuda().eval()
        elif args.surrogate_model == "robust_res50":
            print('--load robust trained res50--')
            surrogate_checkpoint = torch.load(args.robust_res50_path)
            clf = models.__dict__['resnet50']()

            state_dict = surrogate_checkpoint['state_dict']
            new_state_dict = OrderedDict()

            for k, v in state_dict.items():
                k = k.replace('module.', '')
                new_state_dict[k] = v
            
            clf.load_state_dict(new_state_dict)
            clf = clf.cuda()
        
        requires_grad_(clf, False)

        criterion = CrossEntropyLoss(size_average=None, reduce=None, reduction='mean').cuda()
        best_acc = 0

        clf = set_gpu(args, clf)
        clf.cuda()



    # ------------------------------------ step 3/5 : Training --------------------------------------------------
    writer = SummaryWriter(args.outdir)
    test_data_method_model = dict()
    # lr_policy = cosine_lr(optimizer, args)
    for epoch in range(starting_epoch, args.epochs):
        # lr_policy(epoch, iteration=None)
        before = time.time()
        before1 = time.process_time()
        if args.objective == 'denoising':
            if args.surrogate_model == 'incptv3':
                train_loss = train(train_loader, denoiser, criterion, optimizer, epoch, clf, True)
            else:
                train_loss = train(train_loader, denoiser, criterion, optimizer, epoch, clf, False)
            # test_loss = test(test_loader, denoiser, criterion, 0.0, args.print_freq, args.outdir)
            writer.add_scalar('train_loss', train_loss, epoch)
            test_loss = 1000.0
            test_acc = 'NA'


        scheduler.step()
        args.lr = scheduler.get_last_lr()[0]

        # Switch from Adam to SGD
        if epoch == args.start_sgd_epoch and args.optimizer == 'AdamThenSGD ':  # Do adam for few steps thaen continue SGD
            print("-->[Switching from Adam to SGD.]")
            args.lr = args.start_sgd_lr
            optimizer = SGD(denoiser.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
            scheduler = StepLR(optimizer, step_size=args.lr_step_size, gamma=args.gamma)

        torch.save({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': denoiser.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, os.path.join(args.outdir, 'Aug'+str(args.MAEorMSE)+str(args.l1orNOl1)+'mu'+str(args.mu1)+'gamma'+str(args.gamma1)+'lambda'+str(args.lambda1) + 'surrogate' + str(args.surrogate_model) + 'attack_method' + str(args.attack_method) + 'victim_model'+ str(args.victim_model) +'checkpoint.pth.tar'))
        print('save model at {}'.format(args.outdir, 'Aug'+str(args.MAEorMSE)+str(args.l1orNOl1)+'mu'+str(args.mu1)+'gamma'+str(args.gamma1)+'lambda'+str(args.lambda1) + 'surrogate' + str(args.surrogate_model) + 'attack_method' + str(args.attack_method) + 'victim_model'+ str(args.victim_model) +'checkpoint.pth.tar'))

        # ---------------------------begin validation-----------------------------
        attack_method = list(map(lambda x: str(x), args.attack_method.split(",")))
        victim_model = list(map(lambda x: str(x), args.victim_model.split(",")))

        clf_loss = np.zeros((len(attack_method), len(victim_model)))
        clf_acc = np.zeros((len(attack_method), len(victim_model)))
        clf_acc_perturb = np.zeros((len(attack_method), len(victim_model)))

        model_num = 0

        for i, model_name in enumerate(victim_model): #res18,res50,vgg16,vgg19,incptv3
            attack_num = 0
            for j, attack_name in enumerate(attack_method): #PGD,FGSM,CW
                if epoch == 0:
                # print(model_name,attack_name)
                    text_test_list_method_model =  os.path.join(root, 'test_list_{}{}.txt'.format(attack_name,model_name))
                    img_test_clean =  advdata_dir + attack_name + model_name + '/testclean'
                    img_test_adv =  advdata_dir + attack_name + model_name + '/test'
                    tt.gen_txt(text_test_list_method_model, img_test_clean, img_test_adv)
                    test_data_method_model[model_name+attack_name] = RD.FaceDataset(text_test_list_method_model)
                test_data = test_data_method_model[model_name+attack_name]
                if model_name == "incptv3":
                    clf_gt = nn.Sequential(norm_layer,models.inception_v3(pretrained=True)).cuda().eval()
                elif model_name == "vgg16":
                    clf_gt = nn.Sequential(norm_layer, models.vgg16(pretrained=True)).cuda().eval()
                elif model_name == "vgg19":
                    clf_gt = nn.Sequential(norm_layer, models.vgg19(pretrained=True)).cuda().eval()
                elif model_name == "res18":
                    clf_gt = nn.Sequential(norm_layer,models.resnet18(pretrained=True)).cuda().eval()
                elif model_name == "res50":
                    clf_gt = nn.Sequential(norm_layer,models.resnet50(pretrained=True)).cuda().eval()
                
                clf_gt = set_gpu(args, clf_gt)
                clf_gt.cuda()

                classification_criterion = CrossEntropyLoss(size_average=None, reduce=None, reduction = 'mean').cuda()
                test_loader = DataLoader(dataset=test_data, batch_size=64, num_workers=args.workers, pin_memory=False)
                if model_name == "incptv3":
                    # input_batch = input_tensor.unsqueeze(0)
                    clf_loss[attack_num,model_num], clf_acc[attack_num,model_num], clf_acc_perturb[attack_num,model_num] = test_with_classifier(test_loader, denoiser, classification_criterion, args.noise_sd, args.print_freq, clf_gt,clf_gt,True, args.outdir)
                else:
                    clf_loss[attack_num,model_num], clf_acc[attack_num,model_num], clf_acc_perturb[attack_num,model_num] = test_with_classifier(test_loader, denoiser, classification_criterion, args.noise_sd, args.print_freq, clf_gt,clf_gt,False, args.outdir)
                print('The average MSE between reconstructed images and clean images for attack method {} and victim model {} is {}'.format(attack_name,model_name,clf_loss[attack_num,model_num]))
                print('Reconstructed Image Accuracy for attack method {} and victim model {} is {}'.format(attack_name,model_name,clf_acc[attack_num,model_num]))
                print('Reconstructed Adversarial Image Accuracy (compared with the label of the adversarial example) for attack method {} and victim model {} is {}'.format(attack_name,model_name,clf_acc_perturb[attack_num,model_num]))
                attack_num = attack_num + 1
            model_num = model_num + 1

            
        
        clf_loss_avg = np.mean(clf_loss)
        clf_acc_avg = np.mean(clf_acc)
        clf_acc_perturb_avg = np.mean(clf_acc_perturb)
        writer.add_scalar('validate_loss', clf_loss_avg, epoch)
        writer.add_scalar('validate_acc', clf_acc_avg, epoch)
        writer.add_scalar('validate_perturb_acc', clf_acc_perturb_avg, epoch)
        print('The average MSE, clean accuracy, and attack successful rate across the dataset is {}, {}, {}'.format(clf_loss_avg,clf_acc_avg,clf_acc_perturb_avg))
        #-----------------------------------end validation--------------------------------------
        if args.objective == 'denoising' and clf_acc_avg > best_acc:
            best_acc = clf_acc_avg

            torch.save({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': denoiser.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, os.path.join(args.outdir, 'Aug'+str(args.MAEorMSE)+str(args.l1orNOl1)+'mu'+str(args.mu1)+'gamma'+str(args.gamma1)+'lambda'+str(args.lambda1) + 'surrogate' + str(args.surrogate_model) + 'attack_method' + str(args.attack_method) + 'victim_model'+ str(args.victim_model) +'bestpoint.pth.tar'))


        after = time.time()
        after1 = time.process_time()
        if args.objective == 'denoising':
            log(logfilename, "{}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}".format(
                epoch, after - before, after1 - before1,
                args.lr, train_loss, clf_loss_avg, clf_acc_avg, clf_acc_perturb_avg))
        # if args.objective == 'denoising':
        #     log(logfilename, "{}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}".format(
        #         epoch, after - before,
        #         args.lr, train_loss, test_loss, test_acc))



def train(loader: DataLoader, denoiser: torch.nn.Module, criterion, optimizer: Optimizer, epoch: int, classifier: torch.nn.Module, incptv3:bool):
    """
    Function for training denoiser for one epoch
        :param loader:DataLoader: training dataloader
        :param denoiser:torch.nn.Module: the denoiser being trained
        :param criterion: loss function
        :param optimizer:Optimizer: optimizer used during trainined
        :param epoch:int: the current epoch (for logging)
        :param noise_sd:float: the std-dev of the Guassian noise perturbation of the input
        :param classifier:torch.nn.Module=None: a ``freezed'' classifier attached to the denoiser
                                                (required classifciation/stability objectives), None for denoising objective
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()

    MSE = MSELoss(size_average=None, reduce=None, reduction='mean').cuda()
    MAE = L1Loss(size_average=None, reduce=None, reduction='mean').cuda()
    # switch to train mode
    denoiser.train()
    # if classifier:
    #     classifier.eval()
    dict_aug = {'rotate': rotation, 'flip': flip, 'translate': translate, 'cutout': cut_out, 'crop&pad':crop_w_padding, 'cutmix': cut_mix}

    for i, (cleans, advs) in enumerate(loader):
        # measure data loading time
        # data_time.update(time.time() - end)
        batch_size = cleans.shape[0]
        cleans = cleans.cuda().to(dtype=torch.float)
        advs = advs.cuda().to(dtype=torch.float)

        if args.aug != '' and args.aug2 == '':
            cleans_crop, advs_crop = dict_aug[args.aug](cleans, advs)
            batch_size = len(cleans)
            cleans = torch.cat((cleans, cleans_crop), 0)
            advs = torch.cat((advs, advs_crop), 0)
        
        if args.aug2 != '':
            cleans_crop, advs_crop = dict_aug[args.aug](cleans, advs)
            cleans_crop2, advs_crop2 = dict_aug[args.aug2](cleans, advs)
            batch_size = len(cleans)
            cleans = torch.cat((cleans, cleans_crop, cleans_crop2), 0)
            advs = torch.cat((advs, advs_crop, advs_crop2), 0)
        data_time.update(time.time() - end)

        # compute output
        batch_size = len(cleans)
        denoised = denoiser(advs)
        

        if args.eta>0:
            perturbation = advs-cleans
            perturbation = perturbation.view(batch_size,-1)
            reconstruction = advs-denoised
            reconstruction = reconstruction.view(batch_size,-1)
            cos = nn.CosineSimilarity(dim=1, eps=1e-6) 
            simi = cos(perturbation, reconstruction)  # the angle at f_adv 
            mean_simi = torch.mean(simi)

        # adv_outputs = classifier(advs - denoised + cleans)
        # f_denoised = classifier(denoised)
        # clean_label = classifier(cleans)
        # adv_label = classifier(advs)
        
        # clean_label = clean_label.argmax(1).detach().clone()
        # adv_label = adv_label.argmax(1).detach().clone()

            # loss1 = MSE(denoised, cleans)
        if args.MAEorMSE == 'MAE':
            loss1 = MAE(denoised, cleans)
        elif args.MAEorMSE == 'MSE':
            loss1 = MSE(denoised, cleans)
        elif args.MAEorMSE == 'Stability':
            loss1 = 0
        # loss = loss1 - args.eta * mean_simi
        loss = loss1
        # print(MSE(cleans,advs),MSE(cleans,denoised))
        

        if args.gamma1>0:
            f_clean = classifier(cleans)
            F_clean = f_clean.argmax(1).detach().clone()
            f_denoised = classifier(denoised)
            if args.reg_data == 'all':
                loss2 = criterion(f_denoised, F_clean)
            elif args.reg_data =='ori':
                loss2 = criterion(f_denoised[:batch_size], F_clean[:batch_size])
            elif args.reg_data == 'robust':
                F_ori = F_clean[:batch_size]
                F_ori = F_ori.repeat(int(F_clean.shape[0]/batch_size))
                strong_clean = (F_clean == F_ori)
                loss2 = criterion(f_denoised[strong_clean], F_clean[strong_clean])
            loss = loss + args.gamma1 * loss2
        if args.lambda1>0:
            f_adv = classifier(advs)
            F_adv = f_adv.argmax(1).detach().clone()
            f_reon_adv = classifier(advs - denoised + cleans)
            if args.reg_data == 'all':
                loss3 = criterion(f_reon_adv, F_adv)
            elif args.reg_data == 'ori':
                loss3 = criterion(f_reon_adv[:batch_size], F_adv[:batch_size])
            elif args.reg_data == 'robust':
                F_adv_ori = F_adv[:batch_size]
                F_adv_ori = F_adv_ori.repeat(int(F_adv.shape[0]/batch_size))
                strong_adv = (F_adv == F_adv_ori)
                loss3 = criterion(f_reon_adv[strong_adv], F_adv[strong_adv])
            loss = loss + args.lambda1 * loss3


        losses.update(loss.item(), cleans.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                epoch, i, len(loader), batch_time=batch_time,
                data_time=data_time, loss=losses))

    return losses.avg




def frozen_module(module):
    for param in module.parameters():
            param.requires_grad = False

# 1. random flip
def vflip(clean,adv):
    return TF.vflip(clean), TF.vflip(adv)

def hflip(clean,adv):
    return TF.hflip(clean), TF.hflip(adv)

def flip(clean,adv):
    if random.random() > 0.5:
        return vflip(clean,adv)
    else:
        return hflip(clean,adv)

# 2. random rotation
def rotation(clean, adv):
    angle = transforms.RandomRotation.get_params([-180, 180])
    return TF.rotate(clean,angle), TF.rotate(adv,angle)



# def crop_resize(clean, adv):
#     clean_crop = TF.resized_crop(clean, 10, 10, 200, 200, 224)
#     adv_crop = TF.resized_crop(adv, 10, 10, 200, 200, 224)
#     return clean_crop, adv_crop

# 3.  cut out
class Cutout(object):
    def __init__(self, n_holes, length, random=True):
        self.n_holes = n_holes
        self.length = length
        self.random = random

    def __call__(self, img, adv):
        h = img.size(2)
        w = img.size(3)
        length = random.randint(1, self.length)
        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - length // 2, 0, h)
            y2 = np.clip(y + length // 2, 0, h)
            x1 = np.clip(x - length // 2, 0, w)
            x2 = np.clip(x + length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img).cuda()

        img = img * mask
        adv = adv * mask

        return img, adv

def cut_out(clean, adv):
    # crop some area out and fill with zeros
    cutout = Cutout(1, np.random.randint(50)+1)
    return cutout(clean, adv)
    
class Cut_w_padding(object):
    def __init__(self, n_holes, length, random=True):
        self.n_holes = n_holes
        self.length = length
        self.random = random

    def __call__(self, img, adv):
        h = img.size(2)
        w = img.size(3)
        length = random.randint(self.length, h)  # set the random length as the lower bound instead of the upper bound
        mask = np.zeros((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - length // 2, 0, h)
            y2 = np.clip(y + length // 2, 0, h)
            x1 = np.clip(x - length // 2, 0, w)
            x2 = np.clip(x + length // 2, 0, w)

            mask[y1: y2, x1: x2] = 1.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img).cuda()

        img = img * mask
        adv = adv * mask

        return img, adv
    

# 4. crop and pad with zero, as a complement to cut_out
def crop_w_padding(clean, adv):
    # crop some area within the pic and pad the rest with zero-value pixels
    crop = Cut_w_padding(1, np.random.randint(180)+1)
    return crop(clean, adv)

# 5. translate
def translate(clean, adv):
    # translate the pic to some orientation with pixels with pad the rest with zero
    TR = transforms.RandomAffine(0,translate=(0.25,0.25))
    batch_size = len(clean)
    translated = TR(torch.cat((clean, adv), 0))
    return torch.split(translated, batch_size)


# 6. cut and mix 
def cutmix(batch, adv, alpha=1.0): # alpha is control the mixture ratio lambda
    data = batch # since no targets to do denoising, we only need the data itself.
    adv_data = adv

    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_adv_data = adv_data[indices]
    # shuffled_targets = targets[indices] # no targets

    lam = np.random.beta(alpha, alpha)

    image_h, image_w = data.shape[2:]  # BCHW
    cx = np.random.uniform(0, image_w)
    cy = np.random.uniform(0, image_h)
    w = image_w * np.sqrt(1 - lam)
    h = image_h * np.sqrt(1 - lam)
    x0 = int(np.round(max(cx - w / 2, 0)))
    x1 = int(np.round(min(cx + w / 2, image_w)))
    y0 = int(np.round(max(cy - h / 2, 0)))
    y1 = int(np.round(min(cy + h / 2, image_h)))

    data[:, :, y0:y1, x0:x1] = shuffled_data[:, :, y0:y1, x0:x1]
    adv_data[:, :, y0:y1, x0:x1] = shuffled_adv_data[:, :, y0:y1, x0:x1]
    # targets = (targets, shuffled_targets, lam)  # is lam important in our setting? No

    # return both the mixed data and mixed adv_data.
    return data, adv_data

def cut_mix(clean, adv):
    # cut part of the pic out and fill with the corresponding area of another pic.
    return cutmix(clean, adv)




if __name__ == "__main__":
    main()
