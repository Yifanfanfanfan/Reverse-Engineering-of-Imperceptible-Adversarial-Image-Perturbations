import matplotlib.pyplot as plt
import numpy as np
import skimage.measure
import os
from scipy.ndimage.filters import gaussian_filter
import seaborn as sns
import heapq

def plt_densitymap(img,adv,index,name):
    npimg = img.cpu().numpy() 
    npadv = adv.cpu().numpy()
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.axes.get_xaxis().set_visible(False)
    ax1.axes.get_yaxis().set_visible(False)
    perturb = np.abs(npadv-npimg)
    perturb = np_sparsity(perturb)
    perturb_smooth = gaussian_filter(perturb,sigma=1)
    sns.heatmap(perturb_smooth, vmin=np.min(perturb_smooth), vmax=np.max(perturb_smooth), cmap ="coolwarm" , cbar=True )
    if not os.path.isdir('example_visualize/img{}/'.format(index)):
        os.mkdir('example_visualize/img{}/'.format(index))
    plt.savefig('example_visualize/img{}/{}.png'.format(index,name))
    return perturb_smooth

def plt_heatmap(img,adv,index,name):
    npimg = img.cpu().numpy() 
    npadv = adv.cpu().numpy()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    perturb = np.abs(npadv-npimg)
    perturb = np_sparsity(perturb)
    np_min = np.min(perturb)
    np_max = np.max(perturb)
    np_int = np_max-np_min
    im.set_clim(np_min+0.1*np_int,np_max-0.1*np_int)
    cbar = fig.colorbar(ax=ax, mappable=im, orientation='vertical')
    if not os.path.isdir('example_visualize/img{}/'.format(index)):
        os.mkdir('example_visualize/img{}/'.format(index))
    plt.savefig('example_visualize/img{}/{}.png'.format(index,name))

def plt_heatmap_compare(img,denoised,adv,index,name):
    npimg = img.cpu().numpy()
    npdenoised = denoised.cpu().numpy() 
    npadv = adv.cpu().numpy()
    fig = plt.figure()
    ax = fig.add_subplot(121)
    bx = fig.add_subplot(122)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    bx.axes.get_xaxis().set_visible(False)
    bx.axes.get_yaxis().set_visible(False)
    perturb = np.abs(npadv-npimg)
    perturb = np_sparsity(perturb)
    perturb_recon = np.abs(npadv-npdenoised)
    perturb_recon = np_sparsity(perturb_recon)
    im = ax.imshow(perturb,cmap='YlGnBu')
    im_recon = bx.imshow(perturb_recon,cmap='YlGnBu')
    np_min = np.min(perturb)
    np_max = np.max(perturb)
    np_int = np_max-np_min
    np_min_recon = np.min(perturb_recon)
    np_max_recon = np.max(perturb_recon)
    np_int_recon = np_max_recon-np_min_recon
    im.set_clim(np_min+0.1*np_int,np_max-0.1*np_int)
    im_recon.set_clim(np_min_recon+0.1*np_int_recon,np_max-0.1*np_int_recon)
    cbar = fig.colorbar(ax=ax, mappable=im, orientation='vertical')
    cbar = fig.colorbar(ax=bx, mappable=im_recon, orientation='vertical')
    if not os.path.isdir('example_visualize/img{}/'.format(index)):
        os.mkdir('example_visualize/img{}/'.format(index))
    plt.savefig('example_visualize/img{}/{}{}.png'.format(index,name,index))

def np_sparsity(array):
    # reduce image dimension from 3 to 1
    array = (array**2).sum(axis=0)

    array = skimage.measure.block_reduce(array, (2,2), np.sum)
    array = np.sqrt(array)
    # print(array.shape)
    return array

def hist_compare(img,denoised,adv,index,name):
    npimg = img.cpu().numpy().ravel()
    npdenoised = denoised.cpu().numpy().ravel()
    npadv = adv.cpu().numpy().ravel()
    pixel_ind = np.arange(len(npimg))
    perturb_gt = npadv-npimg # a 1-dim flattened perturbation 
    perturb_rec = npadv-npdenoised # a 1-dim flattened reconstruced perturbation
    fig = plt.figure()
    plt.plot(pixel_ind,perturb_gt,label = 'ground truth perturbation')
    plt.plot(pixel_ind,perturb_rec,label = 'reconstructed perturbation')
    plt.xlabel('pixels')
    plt.ylabel('perturbation')
    plt.legend()
    plt.title('groundtruth perturbation vs reconstructed perturbation')
    plt.savefig('example_visualize/img{}/{}_pixelwise.png'.format(index,name))
    plt.close(fig)

def hist_compare_few(img,denoised,adv,index,name):
    npimg = img.cpu().numpy().ravel()
    npdenoised = denoised.cpu().numpy().ravel()
    npadv = adv.cpu().numpy().ravel()
    pixel_ind = np.arange(len(npimg))
    perturb_gt = npadv-npimg # a 1-dim flattened perturbation 
    perturb_rec = npadv-npdenoised # a 1-dim flattened reconstruced perturbation
    fig = plt.figure()
    plt.plot(pixel_ind[0:100],perturb_gt[0:100],label = 'ground truth perturbation')
    plt.plot(pixel_ind[0:100],perturb_rec[0:100],label = 'reconstructed perturbation')
    plt.xlabel('pixels')
    plt.ylabel('perturbation')
    plt.legend()
    plt.title('groundtruth perturbation vs reconstructed perturbation')
    plt.savefig('example_visualize/img{}/{}_pixelwise.png'.format(index,name))
    plt.close(fig)

def distances_l1l2(img1,img2):
    npimg1 = img1.cpu().numpy().ravel()
    npimg2 = img2.cpu().numpy().ravel()
    l1_distance = np.sum(np.abs(npimg1-npimg2))
    l2_distance = np.sqrt(np.square(npimg1-npimg2))

    return l1_distance, l2_distance

def IoU(perturb1,perturb2,top_percent):
    perturb1 = (perturb1.cpu().numpy()**2).sum(axis=0).flatten()
    perturb2 = (perturb2.cpu().numpy()**2).sum(axis=0).flatten()
    # print(np.min(heapq.nlargest(int(len(perturb1)*top_percent),perturb1)))
    mask_perturb1 = perturb1 >= np.min(heapq.nlargest(int(len(perturb1)*top_percent),perturb1))
    mask_perturb2 = perturb2 >= np.min(heapq.nlargest(int(len(perturb2)*top_percent),perturb2))
    I = np.sum(mask_perturb1 * mask_perturb2)
    U = np.sum(mask_perturb1) + np.sum(mask_perturb2) - I
    return I/U

def IoUplot(perturb1,perturb2, img, top_percent):
    perturb1 = (perturb1.cpu().numpy()**2).sum(axis=0).flatten()
    mask_perturb1 = perturb1 >= np.min(heapq.nlargest(int(len(perturb1)*top_percent),perturb1))
    fig = plt.figure()
    plt.imshow(np.transpose(img.cpu().numpy(), (1, 2, 0)))
    plt.imshow(mask_perturb1.reshape(32,32),alpha=0.3)
    plt.savefig('example_visualize/gt_mask.png')
    perturb2 = (perturb2.cpu().numpy()**2).sum(axis=0).flatten()
    mask_perturb2 = perturb2 >= np.min(heapq.nlargest(int(len(perturb2)*top_percent),perturb2))
    fig = plt.figure()
    plt.imshow(np.transpose(img.cpu().numpy(), (1, 2, 0)))
    plt.imshow(mask_perturb2.reshape(32,32),alpha=0.3)
    plt.savefig('example_visualize/recon_mask.png')


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


    
    
