export CUDA_VISIBLE_DEVICES=5

python -u -W ignore test_portion.py         --dataset imagenet\
                                            --denoiser 'Path-to-Pretrained-Denoiser'\
                                            --attack_method FGSM\
                                            --victim_model res18\
                                            --batch 64\
                                            --intensity medium\
                                            --fig_nodes_num 2\