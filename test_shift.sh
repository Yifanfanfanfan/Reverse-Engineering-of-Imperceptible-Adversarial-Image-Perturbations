export CUDA_VISIBLE_DEVICES=3
python -u -W ignore test_denoiser_shift.py        --dataset imagenet\
                                            --denoiser '/home/yifan/github/RED_Denoising/RED_Denoising/code/transform_study/pretrain_no_aug_1.0_portion_300_w_vali/AugMAENOl1mu0.0gamma0.0lambda0.0surrogatevgg19attack_methodPGD,FGSM,CWvictim_modelres18,res50,vgg16,vgg19,incptv3bestpoint.pth.tar'\
                                            --gamma1 0\
                                            --lambda1 0\
                                            --mu1 0\
                                            --reg_data ''\
                                            --attack_method PGD,FGSM,CW\
                                            --victim_model vgg16,vgg19,res18,res50,incptv3\
                                            --batch 16\
                                            --intensity out\
                                            --test_DTx 0\
                                            --test_TDx 0\
                                            # --transform flip,rotation\