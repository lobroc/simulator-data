import numpy as np
import nibabel as nib

from skimage.graph import MCP_Connect
from pathlib import Path

from numpy.typing import NDArray
from typing import Union

def grow_from_seeds(volume_or_image: NDArray, seed_points: NDArray) -> NDArray:
    """
    Performs flood fill on a 3D volume or 2D image from given seed points. Zero values are background and never filled.
    """
    assert seed_points.ndim == 2 and seed_points.shape[1] == volume_or_image.ndim, "Seed points mush be of shape [num_seeds, ndim]"
    assert volume_or_image.ndim in (2, 3), "Only 2D images and 3D volumes are supported."

    flooded = np.zeros_like(volume_or_image, dtype=np.uint8)
    for idx, seed in enumerate(seed_points, start=1):
        flooded[tuple(seed)] = idx
    
    costs = np.where(volume_or_image > 0, 1, np.inf)
    cost_maps = []
    for seed in seed_points:
        mcp = MCP_Connect(costs, fully_connected=True)
        cost, _ = mcp.find_costs(np.expand_dims(seed, axis=0), find_all_ends=True)
        cost_maps.append(cost)
    cost_maps = np.stack(cost_maps, axis=0)
    
    assigned_voxels = (cost_maps.argmin(axis=0) + 1) * (volume_or_image > 0)
    weights = cost_maps.min(axis=0)

    return assigned_voxels, weights

def load_nii(path: Union[Path, str], return_nii_obj: bool = False) -> NDArray:
    vol_obj = nib.load(str(path))
    vol = np.asanyarray(vol_obj.dataobj)
    return (vol, vol_obj) if return_nii_obj else vol

def save_nii(volume: NDArray, path: Union[Path, str], reference_nii_obj: Union[nib.Nifti1Image, None] = None, save_dtype: Union[np.dtype, None] = None) -> None:
    if save_dtype is not None:
        volume = volume.astype(save_dtype)
    else:
        volume = volume.astype(reference_nii_obj.get_data_dtype()) if reference_nii_obj is not None else volume
    
    if reference_nii_obj is not None:
        nib.save(
            nib.Nifti1Image(
                volume,
                affine=reference_nii_obj.affine,
                header=reference_nii_obj.header
            ),
            str(path)
        )
    else:
        affine_mat = np.zeros((4,4))
        affine_mat[:3, :3] = np.eye(3)
        nib.save(
            nib.Nifti1Image(
                volume,
                affine=affine_mat
            ),
            str(path)
        )
