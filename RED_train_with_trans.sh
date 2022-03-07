export CUDA_VISIBLE_DEVICES=4

python3 -u -W ignore RED_train_with_trans.py   --dataset cifar10\
                                            --batch 4\
                                            --arch cifar_dncnn\
                                            --pretrained-denoiser ''\
                                            --surrogate_model 'robust_res50'\
                                            --lr 1e-4\
                                            --epoch 300\
                                            --gamma1 0.025\
                                            --lambda1 0.025\
                                            --mu1 0\
                                            --MAEorMSE MAE\
                                            --l1orNOl1 NOl1\
                                            --attack_method PGD,FGSM,CW\
                                            --victim_model res18,res50,vgg16,vgg19,incptv3\
                                            --lr_step_size 140\
                                            --outdir 'ablation_study/robust_res'\
                                            --data_portion 1.0\
                                            --aug 'crop&pad'\
                                            --aug2 'cutmix'\
                                            --reg_data 'robust'\
                                            # --multigpu 0,1,2,3\
                                            # --transform 'flip,rotation'\