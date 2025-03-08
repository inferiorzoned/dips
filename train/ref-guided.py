# %%
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
from torch.autograd import Variable
import pydicom
from pydicom.dataset import Dataset, FileDataset
import scipy.io as sio

from unet import resunet

# Ensure reproducibility
torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)


# %%
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

    print('initialize network with %s' % init_type)
    net.apply(init_func)

def fft3d_with_shifts(img):
    """3D FFT with shifts"""
    return fft.fftshift(fft.fftn(fft.ifftshift(img), dim=(-3,-2,-1)))

def ifft3d_with_shifts(ksp):
    """3D IFFT with shifts"""
    return fft.fftshift(fft.ifftn(fft.ifftshift(ksp), dim=(-3,-2,-1)))

# def ksp_and_mps_to_gt(ksp, mps):
#     """Convert k-space and sensitivity maps to ground truth image"""
#     gt = mps.conj() * ifft3d_with_shifts(ksp)
#     gt = torch.sum(gt, dim=1)  # Sum across coils
#     return gt

def ksp_and_mps_to_gt(ksp, mps):
    """Convert k-space and sensitivity maps to ground truth image"""
    gt = mps.conj() * ifft3d_with_shifts(ksp)  # (coils, z, y, x)
    gt = torch.sum(gt, dim=0)  # sum across coils, result: (z, y, x)
    return gt

# def mps_and_gt_to_ksp(mps, gt):
#     """Convert sensitivity maps and ground truth to k-space"""
#     ksp = fft3d_with_shifts(mps * gt.unsqueeze(1))  # Add coil dimension
#     return ksp

def mps_and_gt_to_ksp(mps, gt):
    """Convert sensitivity maps and ground truth to k-space"""
    # gt needs to be (z,y,x), mps is (coils,z,y,x)
    gt = gt.unsqueeze(0)  # Add coil dimension at front: (1,z,y,x)
    ksp = fft3d_with_shifts(mps * gt)
    return ksp

def rss_normalize_sense_maps(maps):
    """Normalize 3D sensitivity maps using RSS method"""
    rss = torch.sqrt(torch.sum(torch.abs(maps)**2, dim=1))
    eps = torch.finfo(maps.dtype).eps
    rss = torch.clamp(rss, min=eps)
    normalized_maps = maps / rss.unsqueeze(1)
    return normalized_maps

def safe_normalize(tensor, eps=1e-8):
    """Safely normalize tensor values to [0,1] range"""
    tensor_abs = torch.abs(tensor)
    denominator = tensor_abs.max() - tensor_abs.min()
    if denominator < eps:
        return tensor_abs / (tensor_abs.max() + eps)
    return (tensor_abs - tensor_abs.min()) / denominator

def save_as_dicom_3d(image_data, filename, patient_name="Anonymous"):
    """Save 3D volume as DICOM"""
    file_meta = Dataset()
    file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.4'
    file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
    
    ds = FileDataset(filename, {}, file_meta=file_meta, preamble=b"\0" * 128)
    
    ds.PatientName = patient_name
    ds.PatientID = "123456"
    ds.Modality = "MR"
    ds.ScanningSequence = "RM"
    ds.SequenceVariant = "NONE"
    ds.ScanOptions = "NONE"
    ds.MRAcquisitionType = "3D"
    
    dt = datetime.datetime.now()
    ds.ContentDate = dt.strftime('%Y%m%d')
    ds.ContentTime = dt.strftime('%H%M%S.%f')
    
    image_data = safe_normalize(torch.tensor(image_data)).numpy()
    image_data = (image_data * 65535).astype(np.uint16)
    
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelRepresentation = 0
    ds.Rows, ds.Columns = image_data.shape[:2]
    ds.NumberOfFrames = image_data.shape[2]
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    
    ds.PixelData = image_data.tobytes()
    ds.save_as(filename)

def save_3d_volume_as_dicom(volume_data, output_dir, base_filename):
    """Save 3D volume as series of DICOM files"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Print shape for debugging
    print(f"Volume data shape: {volume_data.shape}")
    
    # Handle different dimensionality cases
    if volume_data.ndim == 4:  # If we have a channel dimension
        volume_data = volume_data[..., 0]  # Take first channel
    
    for z in range(volume_data.shape[2]):
        # Extract slice
        slice_data = volume_data[:, :, z]
        
        # Normalize slice
        slice_min = slice_data.min()
        slice_max = slice_data.max()
        if slice_max > slice_min:  # Avoid division by zero
            slice_data = (slice_data - slice_min) / (slice_max - slice_min)
        slice_data = (slice_data * 65535).astype(np.uint16)
        
        # Create DICOM dataset
        file_meta = pydicom.Dataset()
        file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.4'
        file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
        file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian

        ds = pydicom.Dataset()
        ds.file_meta = file_meta
        ds.is_little_endian = True
        ds.is_implicit_VR = False

        ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
        ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
        ds.StudyInstanceUID = pydicom.uid.generate_uid()
        ds.SeriesInstanceUID = pydicom.uid.generate_uid()

        ds.Modality = "MR"
        ds.SeriesDescription = "Reconstructed MRI"
        ds.ImageType = ["DERIVED", "SECONDARY", "OTHER"]

        ds.Rows, ds.Columns = slice_data.shape
        ds.ImagePositionPatient = [0, 0, z]
        ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        ds.PixelSpacing = [1, 1]
        ds.SliceThickness = 1
        ds.SpacingBetweenSlices = 1

        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.PixelRepresentation = 0
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15

        ds.PixelData = slice_data.tobytes()
        
        # Save the file
        ds.save_as(os.path.join(output_dir, f"{base_filename}_slice_{z:03d}.dcm"))

# %%
# Setup configurations
DATA_DIR = '..//data/my_kspace'
sub_name = 'SUB_NAME'
PLT_SAVE_FOLDER = '../figs/dip_results_3d'
current_setting = 'ref-guided'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load data
print("Loading data...")
kspace_3d = loadmat(os.path.join(DATA_DIR, sub_name, 'renewed/kSpace_forDIP_withPCA_fulltime.mat'))['kSpace']
print(f'K-space shape: {kspace_3d.shape}')
kspace_3d = np.transpose(kspace_3d, (3, 2, 1, 0))  # Reorder to (coils, z, y, x)
print(f'K-space shape after reordering: {kspace_3d.shape}')

with h5py.File(os.path.join(DATA_DIR, sub_name, 'renewed/S_all_fit_forDIP_withPCA_fulltime.mat'), 'r') as file:
    sens_maps = file['S_all_fit'][:]
    sens_maps = sens_maps['real'][()] + 1j * sens_maps['imag'][()]
print(f'Sensitivity maps shape: {sens_maps.shape}')
# For sensitivity maps - change from (z, coils, y, x) to (coils, z, y, x)
sens_maps = np.transpose(sens_maps, (1, 0, 2, 3))
print(f'Sensitivity maps shape after reordering: {sens_maps.shape}')

mask_3d = loadmat(os.path.join(DATA_DIR, sub_name, 'renewed/g_mask_k_space_sparse.mat'))['g_mask_k_space_sparse']
print(f'Mask shape: {mask_3d.shape}')
mask_3d = np.transpose(mask_3d, (2, 1, 0))  # Reorder to match k-space
print(f'Mask shape after reordering: {mask_3d.shape}')

# %%
kspace_tensor = torch.tensor(kspace_3d, dtype=torch.complex64).to(device)
smap_tensor = torch.tensor(sens_maps, dtype=torch.complex64).to(device)
mask_tensor = torch.tensor(mask_3d, dtype=torch.float32).to(device)

# visualize to check
rss_gt_img = ifft3d_with_shifts(kspace_tensor)
rss_gt_img = torch.sqrt(torch.sum(torch.abs(rss_gt_img)**2, dim=0))
sens_gt_img = ksp_and_mps_to_gt(kspace_tensor, smap_tensor)

masked_ksp = mask_tensor * kspace_tensor
rss_masked_gt_img = ifft3d_with_shifts(masked_ksp)
rss_masked_gt_img = torch.sqrt(torch.sum(torch.abs(rss_masked_gt_img)**2, dim=0))
sens_masked_gt_img = ksp_and_mps_to_gt(masked_ksp, smap_tensor)

# vis all the middle slices of the images in a 2 * 2 grid
plt.figure(figsize=(16, 10))
plt.subplot(221)
plt.imshow(sens_gt_img[sens_gt_img.shape[0]//2].T.abs().cpu().numpy(), cmap='gray')
plt.title('Sensitivity GT')
plt.subplot(222)
plt.imshow(rss_gt_img[rss_gt_img.shape[0]//2].T.cpu().numpy(), cmap='gray')
plt.title('RSS GT')
plt.subplot(223)
plt.imshow(sens_masked_gt_img[sens_masked_gt_img.shape[0]//2].T.abs().cpu().numpy(), cmap='gray')
plt.title('Sensitivity Masked GT')
plt.subplot(224)
plt.imshow(rss_masked_gt_img[rss_masked_gt_img.shape[0]//2].T.cpu().numpy(), cmap='gray')
plt.title('RSS Masked GT')
plt.show()

# %%
def print_gpu_memory():
    print(f"Allocated: {torch.cuda.memory_allocated(3)/1e9:.2f}GB")
    print(f"Cached: {torch.cuda.memory_reserved(3)/1e9:.2f}GB")
    
# Call this function at key points in your code
print_gpu_memory()

# Setup configurations
DATA_DIR = '../data/my_kspace'
sub_name = 'SUB_NAME'
PLT_SAVE_FOLDER = '../figs/dip_results_3d'
current_setting = 'ref-guided'
# current_setting = 'test-memory'

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load data
print("Loading data...")
kspace_3d = loadmat(os.path.join(DATA_DIR, sub_name, 'renewed/kSpace_forDIP_withPCA_fulltime.mat'))['kSpace']
print(f'K-space shape: {kspace_3d.shape}')
kspace_3d = np.transpose(kspace_3d, (3, 2, 1, 0))  # Reorder to (coils, z, y, x)
print(f'K-space shape after reordering: {kspace_3d.shape}')

with h5py.File(os.path.join(DATA_DIR, sub_name, 'renewed/S_all_fit_forDIP_withPCA_fulltime.mat'), 'r') as file:
    sens_maps = file['S_all_fit'][:]
    sens_maps = sens_maps['real'][()] + 1j * sens_maps['imag'][()]
print(f'Sensitivity maps shape: {sens_maps.shape}')
# For sensitivity maps - change from (z, coils, y, x) to (coils, z, y, x)
sens_maps = np.transpose(sens_maps, (1, 0, 2, 3))
print(f'Sensitivity maps shape after reordering: {sens_maps.shape}')

mask_3d = loadmat(os.path.join(DATA_DIR, sub_name, 'renewed/g_mask_k_space_sparse_halftime.mat'))['g_mask_k_space_sparse']
print(f'Mask shape: {mask_3d.shape}')
mask_3d = np.transpose(mask_3d, (2, 1, 0))  # Reorder to match k-space
print(f'Mask shape after reordering: {mask_3d.shape}')

# Convert to torch tensors
kspace_tensor = torch.tensor(kspace_3d, dtype=torch.complex64).to(device)
sens_maps_tensor = torch.tensor(sens_maps, dtype=torch.complex64).to(device)
mask_tensor = torch.tensor(mask_3d, dtype=torch.complex64).to(device)

print_gpu_memory()

# Initialize network
net = resunet(n_channels=2, n_classes=2).to(device)
init_weights(net, init_type='normal', init_gain=0.02)

# Training parameters
# no_of_epochs = 4000
no_of_epochs = 2500
learning_rate = 3e-4
show_every = 200

# Initialize input tensor
nz, ny, nx = kspace_tensor.shape[1:]
# ref = Variable(torch.rand((1, 2, nz, ny, nx)).to(device), requires_grad=True)
new_ref = ksp_and_mps_to_gt(mask_tensor * kspace_tensor, sens_maps_tensor)
ref = torch.zeros((1, 2, nz, ny, nx), device=device)
ref[:,0, ...] = new_ref.real
ref[:,1, ...] = new_ref.imag


print_gpu_memory()

# Scale factor computation
with torch.no_grad():
    scale_factor = torch.linalg.norm(net(ref)) / torch.linalg.norm(
        ksp_and_mps_to_gt(kspace_tensor, sens_maps_tensor))
    target_ksp = scale_factor * kspace_tensor
    print('K-space scaled by: ', scale_factor)

# Optimizers
optimizer = optim.Adam(net.parameters(), lr=learning_rate)
# optimizer2 = optim.Adam([ref], lr=1e-1)

# Training loop
# alpha = 2
alpha = 4
exp_weight = 0.99

losses = []
psnrs = []
avg_psnrs = []
# reg_losses = []
fidelity_losses = []

# out_avg = torch.zeros_like(ksp_and_mps_to_gt(target_ksp, sens_maps_tensor)).to(device)
gt = ksp_and_mps_to_gt(target_ksp, sens_maps_tensor)  # (z,y,x)
out_avg = torch.zeros_like(gt.abs()).to(device)  # (z,y,x)
print(f'out_avg shape: {out_avg.shape}')

plt_save_path = os.path.join(PLT_SAVE_FOLDER, sub_name, current_setting)
os.makedirs(plt_save_path, exist_ok=True)

print("Starting training...")
for epoch in tqdm(range(no_of_epochs)):
    optimizer.zero_grad()
    # optimizer2.zero_grad()

    # for vanilla/ ref-guided
    net_output = net(ref)
    net_output_final = torch.view_as_complex(
        net_output.squeeze().permute(1, 2, 3, 0).contiguous())
    # for vanilla/ ref-guided
    
    # print_gpu_memory()
    
    
    # Calculate loss
    pred_ksp = mps_and_gt_to_ksp(sens_maps_tensor, net_output_final)
    # fidelity_loss = torch.linalg.norm(
    #     mask_tensor * fft.fftshift(pred_ksp) - 
    #     mask_tensor * fft.fftshift(target_ksp))
    fidelity_loss = torch.linalg.norm(
        mask_tensor * pred_ksp - 
        mask_tensor * target_ksp)

    loss = fidelity_loss

    losses.append(loss.item())
    # reg_losses.append(reg_loss.item())
    fidelity_losses.append(fidelity_loss.item())
    
    loss.backward()
    optimizer.step()
    # optimizer2.step()
    
    # Track metrics and save results
    with torch.no_grad():
        out = torch.abs(net_output_final)
        out_avg = exp_weight * out_avg + (1 - exp_weight) * out

        # Compute PSNR
        psnr = compute_psnr(gt.abs().cpu().numpy(), out.cpu().numpy(), data_range=gt.abs().max().cpu().numpy())
        psnrs.append(psnr)
        
        if (epoch + 1) % show_every == 0:
            # Save visualizations
            epoch_path = os.path.join(plt_save_path, f'epoch_{epoch}')
            os.makedirs(epoch_path, exist_ok=True)
            
            # Save center slices
            print(f'out shape: {out.shape}')
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

            # save center slices of the ground truth
            gt_view = gt.abs()
            print(f'gt_view shape: {gt_view.shape}')
            plt.figure(figsize=(15, 5))
            plt.subplot(131)
            plt.imshow(gt_view[nz//2].cpu(), cmap='gray')
            plt.title('Axial')
            plt.subplot(132)
            plt.imshow(gt_view[:,ny//2].cpu(), cmap='gray')
            plt.title('Sagittal')
            plt.subplot(133)
            plt.imshow(gt_view[:,:,nx//2].cpu(), cmap='gray')
            plt.title('Coronal')
            plt.savefig(os.path.join(epoch_path, 'gt.png'))
            plt.close()

            # visuzalize the center slices of ref
            ref_view_with_channel = ref.squeeze().permute(1, 2, 3, 0).abs()
            #  take the magnitude of the complex tensor
            ref_view = torch.sqrt(ref_view_with_channel[...,0]**2 + ref_view_with_channel[...,1]**2)
            plt.figure(figsize=(15, 5))
            plt.subplot(131)
            plt.imshow(ref_view[nz//2].cpu(), cmap='gray')
            plt.title('Axial')
            plt.subplot(132)
            plt.imshow(ref_view[:,ny//2].cpu(), cmap='gray')
            plt.title('Sagittal')
            plt.subplot(133)
            plt.imshow(ref_view[:,:,nx//2].cpu(), cmap='gray')
            plt.title('Coronal')
            plt.savefig(os.path.join(epoch_path, 'ref.png'))
            plt.close()

            # visualize the masked gt (ifft)
            print(f'target ksp shape: {target_ksp.shape}')
            print(f'mask tensor shape: {mask_tensor.shape}')    
            mask_tensor_broadcasted = mask_tensor.unsqueeze(0)
            # masked_ksp = mask_tensor_broadcasted * fft.fftshift(target_ksp)
            masked_ksp = mask_tensor_broadcasted * target_ksp
            print(f'masked ksp shape: {masked_ksp.shape}') # 7, 64, 409, 134
            # 7 is the number of coils, 64 is the number of slices, 409 is the number of rows, 134 is the number of columns
            # perform a zero-filled reconstruction
            masked_gt = ifft3d_with_shifts(masked_ksp)
            masked_gt_view = torch.sqrt(torch.sum(masked_gt.abs()**2, dim=0))
            plt.figure(figsize=(15, 5))
            plt.subplot(131)
            plt.imshow(masked_gt_view[nz//2].cpu(), cmap='gray')
            plt.title('Axial')
            plt.subplot(132)
            plt.imshow(masked_gt_view[:,ny//2].cpu(), cmap='gray')
            plt.title('Sagittal')
            plt.subplot(133)
            plt.imshow(masked_gt_view[:,:,nx//2].cpu(), cmap='gray')
            plt.title('Coronal')
            plt.savefig(os.path.join(epoch_path, 'masked_gt.png'))
            plt.close()

            # visualize the masked gt (with sensitivity maps)
            masked_gt_with_smaps = ksp_and_mps_to_gt(masked_ksp, sens_maps_tensor)
            masked_gt_with_smaps_view = masked_gt_with_smaps.abs()
            plt.figure(figsize=(15, 5))
            plt.subplot(131)
            plt.imshow(masked_gt_with_smaps_view[nz//2].cpu(), cmap='gray')
            plt.title('Axial')
            plt.subplot(132)
            plt.imshow(masked_gt_with_smaps_view[:,ny//2].cpu(), cmap='gray')
            plt.title('Sagittal')
            plt.subplot(133)
            plt.imshow(masked_gt_with_smaps_view[:,:,nx//2].cpu(), cmap='gray')
            plt.title('Coronal')
            plt.savefig(os.path.join(epoch_path, 'masked_gt_with_smaps.png'))
            plt.close()

            
            # Save metrics
            plt.figure()
            plt.plot(losses)
            plt.title('Loss')
            plt.savefig(os.path.join(epoch_path, 'loss.png'))
            plt.close()

            # plt.figure()
            # plt.plot(reg_losses)
            # plt.title('Reg Loss')
            # plt.savefig(os.path.join(epoch_path, 'reg_loss.png'))
            # plt.close()

            plt.figure()
            plt.plot(fidelity_losses)
            plt.title('Fidelity Loss')
            plt.savefig(os.path.join(epoch_path, 'fidelity_loss.png'))
            plt.close()

            # combine 3 plots in one
            # plt.figure()
            # plt.plot(losses, label='Total Loss')
            # # plt.plot(reg_losses, label='Reg Loss')
            # plt.plot(fidelity_losses, label='Fidelity Loss')
            # plt.legend()
            # plt.title('Losses')
            # plt.savefig(os.path.join(epoch_path, 'all_losses.png'))
            # plt.close()

            plt.figure()
            plt.plot(psnrs)
            plt.title('PSNR')
            plt.savefig(os.path.join(epoch_path, 'psnr.png'))
            plt.close()
            
            # Save as DICOM
            # if epoch >= 10:
            #     save_as_dicom_3d(
            #         out.cpu().numpy(),
            #         os.path.join(epoch_path, 'reconstruction.dcm')
            #     )
            #     save_as_dicom_3d(
            #         out_avg.cpu().numpy(),
            #         os.path.join(epoch_path, 'reconstruction_avg.dcm')
            #     )
            #     # save_as_dicom_3d(
            #     #     gt.abs().cpu().numpy(),
            #     #     os.path.join(epoch_path, 'gt.dcm')
            #     # )
            #     # save masked
            #     # save_as_dicom_3d(
            #     #     masked_gt_view.cpu().numpy(),
            #     #     os.path.join(epoch_path, 'masked_gt.dcm')
            #     # )
            #     masked_gt_view_permute = masked_gt_view.permute(2, 1, 0).cpu().numpy()
            #     save_3d_volume_as_dicom(
            #         masked_gt_view_permute,
            #         f'{epoch_path}/masked_gt_slices',
            #         'masked-gt-3d'
            #     )
            #     gt_view_permute = gt.abs().permute(2, 1, 0).cpu().numpy()
            #     save_3d_volume_as_dicom(
            #         gt_view_permute,
            #         f'{epoch_path}/gt_slices',
            #         'gt-3d'
            #     )

            # save as 3d matrix
            if epoch >= 400:
                np.save(os.path.join(epoch_path, 'reconstruction.npy'), out.cpu().numpy())
                np.save(os.path.join(epoch_path, 'reconstruction_avg.npy'), out_avg.cpu().numpy())
                np.save(os.path.join(epoch_path, 'gt.npy'), gt.abs().cpu().numpy())
                np.save(os.path.join(epoch_path, 'masked_gt.npy'), masked_gt_view.cpu().numpy())
                np.save(os.path.join(epoch_path, 'masked_gt_with_smaps.npy'), masked_gt_with_smaps_view.cpu().numpy())
        
    # Define header names
    header = 'loss,reg_loss,fidelity_loss'

    # Save losses with header
    np.savetxt(
        os.path.join(plt_save_path, 'losses.csv'),
        # np.array([losses, reg_losses, fidelity_losses]).T,
        np.array([losses, fidelity_losses]).T,
        delimiter=',',
        header=header,  # Add header
    )
    # save all 3 losses in a csv file
    # np.savetxt(os.path.join(plt_save_path, 'losses.csv'), np.array([losses, reg_losses, fidelity_losses]).T, delimiter=',')

print("Training completed!")
