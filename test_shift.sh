export CUDA_VISIBLE_DEVICES=3
python -u -W ignore test_denoiser_shift.py        --dataset imagenet\
                                            --denoiser ''\
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