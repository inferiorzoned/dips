import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.fft as fft
import torch.nn as nn
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
import os
import h5py
from scipy.io import loadmat
import datetime

from unet import resunet

torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)

def init_weights(net, init_type='normal', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, gain=init_gain)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm3d') != -1:
            nn.init.normal_(m.weight.data, 1.0, init_gain)
            nn.init.constant_(m.bias.data, 0.0)
    net.apply(init_func)

def fft3d_with_shifts(img):
    return fft.fftshift(fft.fftn(fft.ifftshift(img), dim=(-3,-2,-1)))

def ifft3d_with_shifts(ksp):
    return fft.fftshift(fft.ifftn(fft.ifftshift(ksp), dim=(-3,-2,-1)))

def ksp_and_mps_to_gt(ksp, mps):
    gt = mps.conj() * ifft3d_with_shifts(ksp)
    gt = torch.sum(gt, dim=0)
    return gt

def mps_and_gt_to_ksp(mps, gt):
    gt = gt.unsqueeze(0)
    ksp = fft3d_with_shifts(mps * gt)
    return ksp

DATA_DIR = '../data/my_kspace'
sub_name = 'SUB_NAME'
PLT_SAVE_FOLDER = '../figs/dip_results_3d'
current_setting = 'aseqdip-3d'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print("Loading data...")
kspace_3d = loadmat(os.path.join(DATA_DIR, sub_name, 'renewed/kSpace_forDIP_withPCA_fulltime.mat'))['kSpace']
kspace_3d = np.transpose(kspace_3d, (3, 2, 1, 0))

with h5py.File(os.path.join(DATA_DIR, sub_name, 'renewed/S_all_fit_forDIP_withPCA_fulltime.mat'), 'r') as file:
    sens_maps = file['S_all_fit'][:]
    sens_maps = sens_maps['real'][()] + 1j * sens_maps['imag'][()]
sens_maps = np.transpose(sens_maps, (1, 0, 2, 3))

mask_3d = loadmat(os.path.join(DATA_DIR, sub_name, 'renewed/g_mask_k_space_sparse.mat'))['g_mask_k_space_sparse']
mask_3d = np.transpose(mask_3d, (2, 1, 0))

kspace_tensor = torch.tensor(kspace_3d, dtype=torch.complex64).to(device)
sens_maps_tensor = torch.tensor(sens_maps, dtype=torch.complex64).to(device)
mask_tensor = torch.tensor(mask_3d, dtype=torch.complex64).to(device)

net = resunet(n_channels=2, n_classes=2).to(device)
init_weights(net, init_type='normal', init_gain=0.02)

no_of_epochs = 2500
learning_rate = 3e-4
show_every = 200

nz, ny, nx = kspace_tensor.shape[1:]
new_ref = ksp_and_mps_to_gt(mask_tensor * kspace_tensor, sens_maps_tensor)
ref = torch.zeros((1, 2, nz, ny, nx), device=device)
ref[:,0, ...] = new_ref.real
ref[:,1, ...] = new_ref.imag

with torch.no_grad():
    scale_factor = torch.linalg.norm(net(ref)) / torch.linalg.norm(
        ksp_and_mps_to_gt(kspace_tensor, sens_maps_tensor))
    target_ksp = scale_factor * kspace_tensor

optimizer = optim.Adam(net.parameters(), lr=learning_rate)

losses = []
psnrs = []
reg_losses = []
fidelity_losses = []

gt = ksp_and_mps_to_gt(target_ksp, sens_maps_tensor)

plt_save_path = os.path.join(PLT_SAVE_FOLDER, sub_name, current_setting)
os.makedirs(plt_save_path, exist_ok=True)

print("Starting training...")
for epoch in tqdm(range(no_of_epochs)):
    
    for i in range(10):
        optimizer.zero_grad()
        
        net_output = net(ref).squeeze()
        net_output_final = torch.view_as_complex(
            net_output.permute(1, 2, 3, 0).contiguous())
        
        pred_ksp = mps_and_gt_to_ksp(sens_maps_tensor, net_output_final)
        
        fidelity_loss = torch.linalg.norm(
            mask_tensor * pred_ksp - mask_tensor * target_ksp)
        reg_loss = torch.linalg.norm(ref - net_output_final)
        loss = fidelity_loss + reg_loss
        
        loss.backward()
        optimizer.step()
    
    with torch.no_grad():
        new_pred_ksp = (1 - mask_tensor) * pred_ksp.detach() / scale_factor + mask_tensor * kspace_tensor
        new_ref = ksp_and_mps_to_gt(new_pred_ksp, sens_maps_tensor)
        ref[:, 0, ...] = new_ref.real
        ref[:, 1, ...] = new_ref.imag

    losses.append(loss.item())
    reg_losses.append(reg_loss.item())
    fidelity_losses.append(fidelity_loss.item())
    
    with torch.no_grad():
        out = torch.abs(net_output_final)
        psnr = compute_psnr(gt.abs().cpu().numpy(), out.cpu().numpy(), 
                           data_range=gt.abs().max().cpu().numpy())
        psnrs.append(psnr)
        
        if (epoch + 1) % show_every == 0:
            epoch_path = os.path.join(plt_save_path, f'epoch_{epoch}')
            os.makedirs(epoch_path, exist_ok=True)
            
            plt.figure(figsize=(15, 5))
            plt.subplot(131)
            plt.imshow(out[nz//2].cpu(), cmap='gray')
            plt.title('Axial')
            plt.subplot(132)
            plt.imshow(out[:,ny//2].cpu(), cmap='gray')
            plt.title('Sagittal')
            plt.subplot(133)
            plt.imshow(out[:,:,nx//2].cpu(), cmap='gray')
            plt.title('Coronal')
            plt.savefig(os.path.join(epoch_path, 'reconstruction.png'))
            plt.close()

            plt.figure()
            plt.plot(losses, label='Total Loss')
            plt.plot(reg_losses, label='Reg Loss')
            plt.plot(fidelity_losses, label='Fidelity Loss')
            plt.legend()
            plt.savefig(os.path.join(epoch_path, 'all_losses.png'))
            plt.close()

            plt.figure()
            plt.plot(psnrs)
            plt.title('PSNR')
            plt.savefig(os.path.join(epoch_path, 'psnr.png'))
            plt.close()
            
            if epoch >= 400 and (epoch + 1) % 400 == 0:
                np.save(os.path.join(epoch_path, 'reconstruction.npy'), out.cpu().numpy())
                np.save(os.path.join(epoch_path, 'gt.npy'), gt.abs().cpu().numpy())
        
    np.savetxt(os.path.join(plt_save_path, 'losses.csv'), 
              np.array([losses, reg_losses, fidelity_losses]).T, 
              delimiter=',', header='loss,reg_loss,fidelity_loss')

print("Training completed!")