#!/usr/bin/env python
# coding: utf-8

import numpy as np
import json5

from sys import path
from pathlib import Path
from skimage.filters import sobel
from argparse import ArgumentParser
from scipy.spatial import ConvexHull, Delaunay
from skimage.morphology import remove_small_objects, ball, erosion

from typing import Tuple, Dict
from numpy.typing import NDArray

path.append(str(Path('.').resolve()))

from align_deformations_to_volume import VolSegAndDeformLoader, load_deformation_field

deformation_aligner = lambda x: np.transpose(x, (1, 2, 0))[:, ::-1, :]
SEGMENTATIONS_DIR = Path('segmentation/')

def get_totalseg_ids() -> Dict[int, str]:

    with open(SEGMENTATIONS_DIR /'mr_segmentation_id_to_name.jsonc', 'r') as f:
        general_totalsegmentator_map = json5.load(f)
    return {int(k): v for k, v in general_totalsegmentator_map.items()}

def create_mri_phantoms(volume_t0: NDArray, airways_mask: NDArray, vessels_mask: NDArray, general_masks: NDArray, lung_mask: NDArray, muscle_mask: NDArray, fat_mask: NDArray, bone_mask: NDArray) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
    general_totalsegmentator_map = get_totalseg_ids()

    general_masks_rot = deformation_aligner(general_masks)
    heart_muscle_id = list(general_totalsegmentator_map.keys())[list(general_totalsegmentator_map.values()).index('heart_muscle')]

    bone_mask, bone_interface_mask = ((bone_mask == 1) | (bone_mask == 2)), (bone_mask == 3) # Rib/spine vs interface

    # Filter to edge of airways
    airways_edges = sobel(airways_mask.astype(float)) > 0 # Magnitude sobel filter

    with open(SEGMENTATIONS_DIR / 'mr_tissue_properties.jsonc', 'r') as f:
        property_map = json5.load(f)

    pd_vol, t1_vol, t2_vol, t2star_vol = np.zeros_like(volume_t0), np.zeros_like(volume_t0), np.zeros_like(volume_t0), np.zeros_like(volume_t0)

    overwrite_add = lambda base, addition: np.where(addition > 0, addition, base)

    # Fat. Largest mask, so add first.
    pd_vol = overwrite_add(pd_vol, property_map['fat']['PD'] * deformation_aligner(fat_mask))
    t1_vol = overwrite_add(t1_vol, property_map['fat']['T1'] * deformation_aligner(fat_mask))
    t2_vol = overwrite_add(t2_vol, property_map['fat']['T2'] * deformation_aligner(fat_mask))
    t2star_vol = overwrite_add(t2star_vol, property_map['fat']['T2*'] * deformation_aligner(fat_mask))

    # Muscle. Also large, comes after.
    pd_vol = overwrite_add(pd_vol, property_map['muscle']['PD'] * deformation_aligner(muscle_mask))
    t1_vol = overwrite_add(t1_vol, property_map['muscle']['T1'] * deformation_aligner(muscle_mask))
    t2_vol = overwrite_add(t2_vol, property_map['muscle']['T2'] * deformation_aligner(muscle_mask))
    t2star_vol = overwrite_add(t2star_vol, property_map['muscle']['T2*'] * deformation_aligner(muscle_mask))

    # Lung tissue. Mask may not be perfect, so add first so that it can be overwritten.
    pd_vol = overwrite_add(pd_vol, property_map['lung']['PD'] * deformation_aligner(lung_mask))
    t1_vol = overwrite_add(t1_vol, property_map['lung']['T1'] * deformation_aligner(lung_mask))
    t2_vol = overwrite_add(t2_vol, property_map['lung']['T2'] * deformation_aligner(lung_mask))
    t2star_vol = overwrite_add(t2star_vol, property_map['lung']['T2*'] * deformation_aligner(lung_mask))

    # Airways
    pd_vol = overwrite_add(pd_vol, property_map['air']['PD'] * deformation_aligner(airways_mask))
    t1_vol = overwrite_add(t1_vol, property_map['air']['T1'] * deformation_aligner(airways_mask))
    t2_vol = overwrite_add(t2_vol, property_map['air']['T2'] * deformation_aligner(airways_mask))
    t2star_vol = overwrite_add(t2star_vol, property_map['air']['T2*'] * deformation_aligner(airways_mask))

    # Airway edges, approximated as cartilage. Not ideal, but better than nothing
    pd_vol = overwrite_add(pd_vol, property_map['cartilage']['PD'] * deformation_aligner(airways_edges))
    t1_vol = overwrite_add(t1_vol, property_map['cartilage']['T1'] * deformation_aligner(airways_edges))
    t2_vol = overwrite_add(t2_vol, property_map['cartilage']['T2'] * deformation_aligner(airways_edges))
    t2star_vol = overwrite_add(t2star_vol, property_map['cartilage']['T2*'] * deformation_aligner(airways_edges))

    # Vessels
    pd_vol = overwrite_add(pd_vol, property_map['blood']['PD'] * deformation_aligner(vessels_mask))
    t1_vol = overwrite_add(t1_vol, property_map['blood']['T1'] * deformation_aligner(vessels_mask))
    t2_vol = overwrite_add(t2_vol, property_map['blood']['T2'] * deformation_aligner(vessels_mask))
    t2star_vol = overwrite_add(t2star_vol, property_map['blood']['T2*'] * deformation_aligner(vessels_mask))

    # Heart muscle
    pd_vol = overwrite_add(pd_vol, property_map['heart_muscle']['PD'] * (general_masks_rot == heart_muscle_id))
    t1_vol = overwrite_add(t1_vol, property_map['heart_muscle']['T1'] * (general_masks_rot == heart_muscle_id))
    t2_vol = overwrite_add(t2_vol, property_map['heart_muscle']['T2'] * (general_masks_rot == heart_muscle_id))
    t2star_vol = overwrite_add(t2star_vol, property_map['heart_muscle']['T2*'] * (general_masks_rot == heart_muscle_id))

    # Bone
    pd_vol = overwrite_add(pd_vol, property_map['bone']['PD'] * bone_mask)
    t1_vol = overwrite_add(t1_vol, property_map['bone']['T1'] * bone_mask)
    t2_vol = overwrite_add(t2_vol, property_map['bone']['T2'] * bone_mask)
    t2star_vol = overwrite_add(t2star_vol, property_map['bone']['T2*'] * bone_mask)

    # Bone interface
    pd_vol = overwrite_add(pd_vol, property_map['bone_interface']['PD'] * bone_interface_mask)
    t1_vol = overwrite_add(t1_vol, property_map['bone_interface']['T1'] * bone_interface_mask)
    t2_vol = overwrite_add(t2_vol, property_map['bone_interface']['T2'] * bone_interface_mask)
    t2star_vol = overwrite_add(t2star_vol, property_map['bone_interface']['T2*'] * bone_interface_mask)

    return pd_vol, t1_vol, t2_vol, t2star_vol

def make_bone_type_volume(general_masks: NDArray, volume_t0: NDArray) -> NDArray:
    general_totalsegmentator_map = get_totalseg_ids()
    erosion_radius = np.cbrt(np.prod(general_masks.shape)) / 100
    footprint = ball(erosion_radius)

    general_masks_rot = deformation_aligner(general_masks)
    rib_bone_ids = [k for k, v in general_totalsegmentator_map.items() if 'rib_bone' == v]
    spine_bone_ids = [k for k, v in general_totalsegmentator_map.items() if 'spine_bone' == v]

    bone_binary_vol = np.zeros_like(volume_t0, dtype=bool)
    bone_binary_vol |= np.isin(general_masks_rot, rib_bone_ids + spine_bone_ids)
    bone_interfaces = (bone_binary_vol.astype(np.uint8) - erosion(bone_binary_vol, footprint).astype(np.uint8)).astype(bool)

    # Make union of all bone masks
    bone_type_vol = np.zeros_like(volume_t0, dtype=np.uint8)
    for bone_id in rib_bone_ids:
        bone_type_vol[general_masks_rot == bone_id] = 1
    for bone_id in spine_bone_ids:
        bone_type_vol[general_masks_rot == bone_id] = 2

    bone_type_vol[bone_interfaces] = 3

    return bone_type_vol

def convex_hull_3d_fast(mask: np.ndarray) -> np.ndarray:
    points = np.argwhere(mask)

    if len(points) > 10_000:
        idx = np.random.choice(len(points), 10_000, replace=False)
        points = points[idx]

    # 2. Compute convex hull (fast, in point space)
    hull = ConvexHull(points)

    # 3. Rasterize: test all voxels against the hull
    grid_points = np.argwhere(np.ones(mask.shape, dtype=bool))
    deln = Delaunay(points[hull.vertices])
    in_hull = deln.find_simplex(grid_points) >= 0

    result = np.zeros(mask.shape, dtype=bool)
    result[tuple(grid_points[in_hull].T)] = True
    return result

if __name__ == "__main__":
    parser = ArgumentParser(description="Apply deformation to lung meshes.")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--dataset-nii-dir", type=str, help="Directory containing the original NIfTI volumes.")
    parser.add_argument("--dataset-segmentations-dir", type=str, help="Directory containing the segmentation NIfTI files.")
    parser.add_argument("--deformation-index", type=int, help="Index of the deformation to use.")
    parser.add_argument("--patient-index", type=int, help="Index of the patient to use.")
    parser.add_argument("--alignment-dir", type=str, help="Directory containing the alignment files.")
    args = parser.parse_args()
    data_base_dir = Path(args.data_dir)
    data_nii_dir = Path(args.dataset_nii_dir)
    data_segmentations_dir = Path(args.dataset_segmentations_dir)
    alignment_dir = Path(args.alignment_dir)

    lung_side = 'left' # Dummy value

    volsegloader = VolSegAndDeformLoader(data_base_dir, data_nii_dir, data_segmentations_dir) # Load the same way as in apply_deform_to_lungs.py
    (_,
        _,
        _,
        lung_segmentation,
        body_segmentation,
        muscle_fat_segmentation,
        airways_mask,
        vessels_mask,
        general_masks # TotalSegmentator passthrough
    ) = volsegloader.load_deform_and_volume(
        deformation_index=args.deformation_index,
        patient_index=args.patient_index,
        lung=lung_side,
        verbose=True,
        return_vessels=True,
        return_general_masks=True,
        deform_as_proportional_shift=False, # Absolute coordinates
    )
    lung_segmentation, body_segmentation, muscle_fat_segmentation, general_masks = lung_segmentation.numpy(), body_segmentation.numpy(), muscle_fat_segmentation.numpy(), general_masks.numpy()
    lung_segmentation = (lung_segmentation > 0).astype(bool)

    relative_deformations, mesh_t0, patient_segmentation, volume_t0 = load_deformation_field(
        base_dir=data_base_dir,
        dataset_nii_dir=data_nii_dir,
        dataset_segmentations_dir=data_segmentations_dir,
        alignment_dir=alignment_dir,
        deformation_index=args.deformation_index,
        patient_index=args.patient_index,
    )

    # Density
    volume_t0 = volume_t0 - np.min(volume_t0) # Shift to [0, max]
    original_volume = (volume_t0 / np.max(volume_t0))[::-1, :, :] # Normalize to [0, 1]

    lung_only_volume = volume_t0 * deformation_aligner(vessels_mask | airways_mask | lung_segmentation)
    body_everything_seg = lung_segmentation | vessels_mask | airways_mask | (general_masks > 0) | (muscle_fat_segmentation > 0)
    body_everything_seg = remove_small_objects(body_everything_seg, max_size=100)
    body_rebuilt_seg = convex_hull_3d_fast(body_everything_seg)

    muscle_infill_mask = np.bitwise_xor(body_rebuilt_seg, body_everything_seg) # Try to in-fill muscle, where there are gaps in the segmentation

    bone_type_vol = make_bone_type_volume(general_masks, volume_t0)

    pd_vol, t1_vol, t2_vol, t2star_vol = create_mri_phantoms(
        volume_t0=lung_only_volume,
        airways_mask=airways_mask,
        vessels_mask=vessels_mask,
        general_masks=general_masks,
        lung_mask=lung_segmentation,
        muscle_mask=(muscle_infill_mask | (muscle_fat_segmentation == 2).astype(bool)),
        fat_mask=(muscle_fat_segmentation == 1).astype(bool),
        bone_mask=bone_type_vol
    )

    # Save data for simulator
    output_dir = data_base_dir / 'simulator-data'
    output_dir.mkdir(exist_ok=True, parents=True)
    output_dir_case = output_dir / f'patient_{args.patient_index}_deformation_{args.deformation_index}'
    output_dir_case.mkdir(exist_ok=True, parents=True)

    np.save(output_dir_case / 't1_volume.npy', t1_vol)
    np.save(output_dir_case / 't2_volume.npy', t2_vol)
    np.save(output_dir_case / 't2star_volume.npy', t2star_vol)
    np.save(output_dir_case / 'mesh_t0.npy', mesh_t0)
    np.save(output_dir_case / 'bone_type_volume.npy', bone_type_vol)
    np.save(output_dir_case / 'relative_deformations.npy', relative_deformations)
    np.save(output_dir_case / 'proton_density_volume.npy', pd_vol)
    np.save(output_dir_case / 'original_volume.npy', original_volume)
