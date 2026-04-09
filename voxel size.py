import nibabel as nib
import numpy as np

def get_voxel_size_and_count(nifti_path):

    img = nib.load(nifti_path)
    affine = img.affine

    voxel_size = np.abs(np.diag(affine)[:3])

    brain_mask_data = img.get_fdata().astype(bool)

    brain_voxel_count = np.count_nonzero(brain_mask_data)

    return voxel_size, brain_voxel_count

brain_mask_path = "newboldmask.nii"
voxel_size, brain_voxel_count = get_voxel_size_and_count(brain_mask_path)

print("voxel size (mm): ", voxel_size)
print("Brain mask voxel count: ", brain_voxel_count)
