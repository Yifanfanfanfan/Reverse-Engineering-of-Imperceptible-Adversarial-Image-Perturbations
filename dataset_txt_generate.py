import os


def gen_txt(txt_path, img_dir_clean, img_dir_adv):
    f = open(txt_path, 'w')
    i = 0
    for s_dirs in sorted(os.listdir(img_dir_clean)):  # 获取 train文件下各文件夹名称
        clean_dir = os.path.join(img_dir_clean, s_dirs)
        adv_dir = os.path.join(img_dir_adv, s_dirs)
        adv_dir_new = adv_dir.replace('clean','S')
        # print(adv_dir_new)
        
        if not clean_dir.endswith('bmp'):
            continue

        line = clean_dir + ' ' + adv_dir_new + ' ' + str(i) + '\n'
        f.write(line)

        i = i + 1

    f.close()

def gen_txt_new(txt_path, img_dir_clean, img_dir_adv):
    f = open(txt_path, 'w')
    i = 0
    for s_dirs in sorted(os.listdir(img_dir_adv)):  # 获取 train文件下各文件夹名称
        adv_dir = os.path.join(img_dir_adv, s_dirs)
        clean_dir = os.path.join(img_dir_clean, s_dirs)
        # adv_dir = os.path.join(img_dir_adv, s_dirs)
        clean_dir_new = clean_dir.replace('_S','_clean')
        clean_dir_new = clean_dir_new.replace('_PGD_','_')
        clean_dir_new = clean_dir_new.replace('_FGSM_','_')
        clean_dir_new = clean_dir_new.replace('_CW_','_')
        # print(adv_dir_new)
        
        if not clean_dir.endswith('bmp'):
            continue

        line = clean_dir_new + ' ' + adv_dir + ' ' + str(i) + '\n'
        f.write(line)

        i = i + 1

    f.close()

def gen_txt_new_new(txt_path, img_dir_clean, img_dir_adv):
    f = open(txt_path, 'w')
    i = 0
    for s_dirs in sorted(os.listdir(img_dir_adv)):  # 获取 train文件下各文件夹名称
        adv_dir = os.path.join(img_dir_adv, s_dirs)
        clean_dir = os.path.join(img_dir_clean, s_dirs)
        # adv_dir = os.path.join(img_dir_adv, s_dirs)
        clean_dir_new = clean_dir.replace('C_','Clean_')
        clean_dir_new = clean_dir_new.replace('_N_vgg16_WB_','_')
        clean_dir_new = clean_dir_new.replace('_N_vgg19_WB_','_')
        clean_dir_new = clean_dir_new.replace('_N_inceptionv3_WB_','_')
        clean_dir_new = clean_dir_new.replace('_N_resnet18_WB_','_')
        clean_dir_new = clean_dir_new.replace('_N_resnet50_WB_','_')
        clean_dir_new = clean_dir_new.replace('_N_lenet5_WB_','_')
        clean_dir_new = clean_dir_new.replace('_S','')
        # print(adv_dir_new)
        
        if not clean_dir.endswith('bmp'):
            continue

        line = clean_dir_new + ' ' + adv_dir + ' ' + str(i) + '\n'
        f.write(line)

        i = i + 1

    f.close()

def gen_txt_auto(txt_path, img_dir_clean, img_dir_adv):
    f = open(txt_path, 'w')
    i = 0
    for s_dirs in sorted(os.listdir(img_dir_adv)):  # 获取 train文件下各文件夹名称
        adv_dir = os.path.join(img_dir_adv, s_dirs)
        clean_dir = os.path.join(img_dir_clean, ('Clean_' + s_dirs))
        # adv_dir = os.path.join(img_dir_adv, s_dirs)
        
        if not clean_dir.endswith('bmp'):
            continue

        line = clean_dir + ' ' + adv_dir + ' ' + str(i) + '\n'
        f.write(line)

        i = i + 1

    f.close()

def gen_txt_robust_auto(txt_path, img_dir_clean, img_dir_adv):
    f = open(txt_path, 'w')
    i = 0
    for s_dirs in sorted(os.listdir(img_dir_clean)):  # 获取 train文件下各文件夹名称
        adv_dir = os.path.join(img_dir_adv, s_dirs)
        clean_dir = os.path.join(img_dir_clean, s_dirs)
        adv_dir_new = adv_dir.replace('_Clean_', '_')
        
        if not clean_dir.endswith('bmp'):
            continue

        line = clean_dir + ' ' + adv_dir_new + ' ' + str(i) + '\n'
        f.write(line)

        i = i + 1

    f.close()

def gen_txt_proGAN(txt_path, img_dir_clean, img_dir_adv):
    f = open(txt_path, 'w')
    i = 0
    for s_dirs in sorted(os.listdir(img_dir_clean)):  # 获取 train文件下各文件夹名称
        adv_dir = os.path.join(img_dir_adv, s_dirs)
        clean_dir = os.path.join(img_dir_clean, s_dirs)

        line = clean_dir + ' ' + adv_dir + ' ' + str(i) + '\n'
        f.write(line)

        i = i + 1

    f.close()

def gen_txt_CRN(txt_path, img_dir_clean, img_dir_adv):
    f = open(txt_path, 'w')
    i = 0
    for s_dirs in sorted(os.listdir(img_dir_adv)):  # 获取 train文件下各文件夹名称
        adv_dir = os.path.join(img_dir_adv, s_dirs)
        clean_dir = os.path.join(img_dir_clean, ('00' + s_dirs.replace('_output', '')))
        line = clean_dir + ' ' + adv_dir + ' ' + str(i) + '\n'
        f.write(line)

        i = i + 1

    f.close()
