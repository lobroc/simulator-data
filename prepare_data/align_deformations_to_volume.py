#!/usr/bin/env python
# coding: utf-8

import ants
import numpy as np

from pathlib import Path
from sys import path, stderr
from typing import Tuple, Union
from numpy.typing import NDArray

from skimage.morphology import closing, opening, ball, binary_erosion, binary_dilation
from scipy import ndimage as ndi
from skimage.morphology import remove_small_objects
from skimage.measure import centroid
from pyvista import UnstructuredGrid
from argparse import ArgumentParser
from skimage.morphology import remove_small_objects
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

path.append(str(Path('.').resolve()))

from utils import load_nii, grow_from_seeds
from prepare_data.volume_processing_utils import convert_pat_deform_to_ants
from segmentation.retrieve_segmentations import load_segmentations_for_img

def load_deformation_field(base_dir: Path, dataset_nii_dir: Path, dataset_segmentations_dir: Path, alignment_dir: Path, patient_index: int, deformation_index: int):
    '''
    Load back the deformation field and mesh points outputted by align_deformations_to_volume.py script.
    '''

    input_dir = base_dir / alignment_dir if (base_dir / alignment_dir).exists() else alignment_dir
    relative_deformations_path = input_dir / f'mesh_relative_deformations_patient_{patient_index:02d}_deform_{deformation_index:02d}.npy'
    mesh_t0_path = input_dir / f'registered_mesh_points_patient_{patient_index:02d}_deform_{deformation_index:02d}.npy'
    patient_segmentation_path = input_dir / f'cleaned_lung_segmentation_patient_{patient_index:02d}.npy'

    relative_deformations = np.load(relative_deformations_path)
    mesh_t0 = np.load(mesh_t0_path)
    patient_segmentation = np.load(patient_segmentation_path)

    vol_loader = VolSegAndDeformLoader(base_dir, dataset_nii_dir, dataset_segmentations_dir)
    body_seg = vol_loader.load_deform_and_volume(
        deformation_index=deformation_index,
        patient_index=patient_index,
        lung='left', # Irrelevant
        verbose=False,
        deform_as_proportional_shift=True # Irrelevant
    )[4]

    volume_t0_path = vol_loader.image_paths[patient_index]
    volume_t0 = load_nii(volume_t0_path, return_nii_obj=False)
    volume_t0 = body_seg.numpy() * volume_t0
    volume_t0 = np.transpose(volume_t0, (1, 2, 0))
    volume_t0 = volume_t0[::-1, ::-1, :] # Re-orient to match mesh coords

    return relative_deformations, mesh_t0, patient_segmentation, volume_t0

class VolSegAndDeformLoader:
    VESSELS_ID = 5
    AIRWAYS_ID = 6

    def __init__(self, base_dir: Path, dataset_nii_dir: Path, dataset_segmentations_dir: Path):
        self.base_dir = base_dir
        # Load warp fields
        self.deform_patients = tuple(Path("prepare_data").glob("??????_??????_*.npy"))
        self.deform_patients = sorted(self.deform_patients) # Ensure consistent ordering

        # Load image pairs + segmentations
        self.dataset_path = dataset_nii_dir
        self.segmentations_dir = dataset_segmentations_dir
        self.image_paths = tuple(self.dataset_path.rglob("*.nii.gz")) + tuple(self.dataset_path.rglob("*.nii"))

    def load_deform_and_volume(
        self,
        deformation_index: int,
        patient_index: int,
        lung: str = "left",
        verbose: bool = True,
        return_vessels: bool = False,
        return_general_masks: bool = False,
        deform_as_proportional_shift: bool = True,
    ):
        name_match_str = 'mesh_' + ('LL' if lung == 'left' else 'RL')
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_deform = executor.submit(
                convert_pat_deform_to_ants,
                [x for x in self.deform_patients if name_match_str in x.stem][deformation_index],
                full_volume=False,
                deform_as_proportional_shift=deform_as_proportional_shift,
            )
            print("Loading segmentations from patient", self.image_paths[patient_index].stem, file=stderr)
            future_segmentations = executor.submit(
                load_segmentations_for_img,
                self.segmentations_dir,
                self.image_paths[patient_index].stem,
            )

            mesh_deformations, _, lung_warp_reference, reference_mesh = future_deform.result()
            segmentations = future_segmentations.result()

        if not all([x.numpy().any() for x in segmentations.values()]):
            raise RuntimeError("One or more segmentations are empty. Please check the segmentation quality.")

        if verbose:
            print("Segmentations loaded:", segmentations.keys(), file=stderr)
        lung_segmentation = segmentations["lungs"]
        body_segmentation = segmentations["body"]
        airways_and_vessels_segmentation = segmentations["airways_and_vessels"]
        muscle_fat_segmentation = segmentations["muscle_fat"]
        airways_mask = airways_and_vessels_segmentation.numpy() == self.AIRWAYS_ID

        # Post-process muscle & fat to group them into representative classes.
        # [0, 1, 2, 3] -> [0, 1 (fat), 2 (muscle)]
        muscle_fat_segmentation = muscle_fat_segmentation.numpy()
        muscle_fat_segmentation[(muscle_fat_segmentation == 1) | (muscle_fat_segmentation == 2)] = 1
        muscle_fat_segmentation[muscle_fat_segmentation == 3] = 2
        muscle_fat_segmentation = ants.from_numpy(muscle_fat_segmentation)

        if lung_segmentation.numpy().sum() < (0.20 * np.prod(lung_segmentation.numpy().shape)): # Should be at least 20% of the volume, otherwise likely an error
            raise RuntimeError("Lung segmentation appears too small. Please check the segmentation quality.")

        if return_vessels:
            vessels_mask = airways_and_vessels_segmentation.numpy() == self.VESSELS_ID

        retlist = [
            mesh_deformations,
            lung_warp_reference,
            reference_mesh,
            lung_segmentation,
            body_segmentation,
            muscle_fat_segmentation,
            airways_mask,
        ]
        if return_vessels:
            retlist.append(vessels_mask)
        if return_general_masks:
            retlist.append(segmentations['general'])

        return tuple(retlist)

class LungSegmentationPostProcessor:
    def __init__(
        self,
        lung_warp_reference: ants.ANTsImage,
        lung_segmentation: ants.ANTsImage,
        airways_mask: NDArray,
    ):
        self.lung_warp_reference = lung_warp_reference
        self.lung_segmentation = lung_segmentation
        self.airways_mask = airways_mask

    @staticmethod
    def partition_segmentation(mask, get_dists_to_zero=False):
        """
        mask  : 3D binary array (1 = foreground)
        This function takes a binary mask of the lungs, and partitions it into left and right lungs.
        """

        middle_slice = mask[mask.shape[0] // 2]

        if not middle_slice.any():
            raise RuntimeError("Middle slice of lung segmentation is empty. Cannot partition lungs. Please check the segmentation quality.")

        middle_slice_left, middle_slice_right = (
            middle_slice[:, : middle_slice.shape[1] // 2],
            middle_slice[:, middle_slice.shape[1] // 2:],
        )
        centroids = [
            centroid(middle_slice_left),
            centroid(middle_slice_right) +
            np.array([0, middle_slice.shape[1] // 2]),
        ]
        centroids = [x.astype(int).tolist() for x in centroids]

        # Create seed image
        seeds = np.zeros(mask.shape, dtype=np.int32)
        c1 = [mask.shape[0] // 2] + centroids[0]
        c2 = [mask.shape[0] // 2] + centroids[1]
        seeds[tuple(c1)] = 1
        seeds[tuple(c2)] = 2

        print("Lung segmentation seed points placed at: ",
                c1, "and", c2, file=stderr)

        # Partition based on closest seed
        labels, distances_to_zero = grow_from_seeds(mask, np.array([c1, c2]))

        return (
            (labels, np.stack([c1, c2]), distances_to_zero)
            if get_dists_to_zero
            else (labels, np.stack([c1, c2]))
        )

    @staticmethod
    def clean_mask(mask: NDArray, strong: float = 0.00) -> NDArray:
        """Cleans a binary mask using morphological operations."""
        original_mask = mask.copy()
        # Pad by 10%
        original_size = mask.shape
        mask = np.pad(
            mask,
            pad_width=[(s // 10, s // 10) for s in mask.shape],
            mode="constant",
            constant_values=0,
        )

        # Morphological operations
        mask = closing(mask.astype(bool), footprint=ball(10, decomposition="sequence")).astype(mask.dtype)
        mask = opening(mask.astype(bool), footprint=ball(5, decomposition="sequence")).astype(mask.dtype)

        # Crop back to original size
        slices = tuple(
            slice(s // 10, s // 10 + os) for s, os in zip(mask.shape, original_size)
        )
        mask = mask[slices]

        if strong > 0:
            # Break small connectors
            mask = binary_erosion(mask.astype(bool), footprint=ball(int(np.max(original_size) * strong), decomposition="sequence")).astype(mask.dtype)
            mask = remove_small_objects(mask.astype(bool), min_size=0.01 * np.prod(original_size)).astype(mask.dtype)
            mask = binary_dilation(mask.astype(bool), footprint=ball(int(np.max(original_size) * strong), decomposition="sequence")).astype(mask.dtype)

            if not mask.any() and strong > 0.01:
                return LungSegmentationPostProcessor.clean_mask(original_mask, strong=strong-0.01) # Retry with more lenient params

        return mask

    def process(self, reorient: bool = False, reference_mesh_to_reorient: Union[UnstructuredGrid, None] = None, deformations_to_reorient: Union[NDArray, None] = None) -> NDArray:
        # Modify volumes so that they are oriented in the same coordinate system
        lung_warp_ref_reoriented = np.transpose(self.lung_warp_reference.numpy(), (2, 0, 1)) if reorient else self.lung_warp_reference.numpy()
        if reference_mesh_to_reorient is not None:
            reference_mesh_to_reorient.points = reference_mesh_to_reorient.points[:, [2, 0, 1]] if reorient else reference_mesh_to_reorient.points
        if deformations_to_reorient is not None:
            deformations_to_reorient = deformations_to_reorient[:, :, [2, 0, 1]] if reorient else deformations_to_reorient

        lung_segmentation_reoriented = self.lung_segmentation.numpy() * (np.ones_like(self.airways_mask) ^ self.airways_mask) # Remove airways
        if reorient:
            lung_segmentation_reoriented = lung_segmentation_reoriented.transpose((1, 2, 0))[:, ::-1, :]

        # Clean up masks, so that they have no more holes. Makes the alignment easier, and less prone to errors.
        checkpoint = lung_warp_ref_reoriented.copy()
        lung_warp_ref_reoriented = self.clean_mask(lung_warp_ref_reoriented, strong=0.00)
        if lung_warp_ref_reoriented.sum() == 0:
            print("Warning: warp reference mask completely eroded during cleaning. Reverting to uncleaned version.", file=stderr)
            lung_warp_ref_reoriented = checkpoint.copy()
        # lung_segmentation_reoriented = self.clean_mask(lung_segmentation_reoriented, strong=0.00)

        # While excluding the background, find two centroids (for each lung) with kmeans
        lung_segmentation_reoriented_with_floodfill, centroids, dist_to_zero = self.partition_segmentation(lung_segmentation_reoriented, get_dists_to_zero=True)
        ncomp = np.unique(lung_segmentation_reoriented_with_floodfill).size
        assert ncomp == 3, f"Expected 3 components after floodfill (bg + 2 lungs), got {ncomp}"

        left_right_axis = np.diff(centroids, axis=0).argmax()
        # Crop max / min in other axes to minimum between two lungs. This ensures that no blobs that were left behind 'survive'.
        nonz_first, nonz_second = [np.stack(np.nonzero(lung_segmentation_reoriented_with_floodfill == i), axis=0) for i in range(1, 3)]
        max_first, min_first = nonz_first.max(axis=1), nonz_first.min(axis=1)
        max_second, min_second = nonz_second.max(axis=1), nonz_second.min(axis=1)

        selected_maxs = np.where(max_first > max_second, max_second, max_first) + (np.abs(max_first - max_second) * 0.05).astype(int)
        selected_mins = np.where(min_first < min_second, min_second, min_first) - (np.abs(min_first - min_second) * 0.05).astype(int)

        # Apply crop
        output_seg = np.zeros_like(lung_segmentation_reoriented_with_floodfill)
        for i in range(1, 3):
            mask = lung_segmentation_reoriented_with_floodfill == i
            for ax in range(3):
                if ax == left_right_axis:
                    continue
                zeroer = (np.arange(mask.shape[ax]) < selected_maxs[ax]) & (np.arange(mask.shape[ax]) > selected_mins[ax])
                mask = np.apply_along_axis(lambda x: x * zeroer, ax, mask)

            output_seg[mask] = i

        to_ret = [
            output_seg,
            centroids,
            lung_warp_ref_reoriented,
        ]
        if reference_mesh_to_reorient is not None:
            to_ret.append(reference_mesh_to_reorient)
        if deformations_to_reorient is not None:
            to_ret.append(deformations_to_reorient)
        return tuple(to_ret)

class AffineTransformer:
    def __init__(self, lung_side: str):
        self.lung_side = lung_side

    def rotate_warp_ref(self, warp_ref: NDArray, lung_segmentation: NDArray) -> NDArray:
        # reor = np.transpose(warp_ref, (1, 0, 2))
        reor = warp_ref.copy()

        # Flip x axis
        reor = reor[::-1, :, :]
        # Flip z axis
        # reor = reor[:, :, ::-1]

        return reor

    def rotate_points(self, reference_mesh_points, deformations: NDArray, lung_segmentation: NDArray) -> NDArray:
        # reor = reference_mesh_points[:, [1, 0, 2]]
        # deformations_reor = deformations[:, :, [1, 0, 2]]

        reor = reference_mesh_points
        deformations_reor = deformations

        # # Flip x axis
        reor[:, 0] = (-1 * (reor[:, 0] - np.mean(reor[:, 0]))) + np.mean(reor[:, 0])
        deformations_reor[:, :, 0] = -1 * deformations_reor[:, :, 0]

        # # Flip z axis
        # reor[:, 2] = (-1 * (reor[:, 2] - np.mean(reor[:, 2]))) + np.mean(reor[:, 2])
        # deformations_reor[:, :, 2] = -1 * deformations_reor[:, :, 2]

        return reor, deformations_reor

    def translate_warp_ref(self, warp_ref: NDArray, lung_segmentation: NDArray) -> NDArray:
        warp_ref_centroid = centroid(warp_ref)
        lung_seg_centroid = centroid(lung_segmentation == (1 if self.lung_side == "left" else 2))
        translation_vector = lung_seg_centroid - warp_ref_centroid

        warp_ref_moved = ndi.shift(warp_ref, shift=translation_vector, order=1, mode='nearest')
        return warp_ref_moved

    def translate_points(self, reference_mesh_points, deformations: NDArray, lung_segmentation: NDArray) -> NDArray:
        warp_ref_centroid = reference_mesh_points.mean(axis=0)
        lung_seg_centroid = centroid(lung_segmentation == (1 if self.lung_side == "left" else 2))
        translation_vector = lung_seg_centroid - warp_ref_centroid

        reor_mesh_points = reference_mesh_points + translation_vector
        reor_deformations = deformations # No change, deliberately!

        return reor_mesh_points, reor_deformations

    def scale_warp_ref(self, warp_ref: NDArray, lung_segmentation: NDArray) -> NDArray:
        warp_ref_extent = np.argwhere(warp_ref > 0)
        lung_seg_extent = np.argwhere(lung_segmentation == (1 if self.lung_side == "left" else 2))

        scale_factors = (lung_seg_extent.max(axis=0) - lung_seg_extent.min(axis=0)) / (warp_ref_extent.max(axis=0) - warp_ref_extent.min(axis=0))

        centered_obj = ndi.shift(warp_ref, shift=(np.array(warp_ref.shape) // 2) - centroid(warp_ref))

        centre = np.array(centered_obj.shape) / 2.0
        offset = centre - np.diag(1/scale_factors) @ centre
        rescaled_warp_ref = ndi.affine_transform(
            centered_obj,
            matrix=np.diag(1/scale_factors),
            offset=offset,
            order=1,
            mode='nearest'
        )
        return rescaled_warp_ref

    def scale_points(self, reference_mesh_points, deformations: NDArray, lung_segmentation: NDArray) -> NDArray:
        lung_seg_extent = np.argwhere(lung_segmentation == (1 if self.lung_side == "left" else 2))

        scale_factors = (lung_seg_extent.max(axis=0) - lung_seg_extent.min(axis=0)) / (reference_mesh_points.max(axis=0) - reference_mesh_points.min(axis=0))

        if (reference_mesh_points.min(axis=0) < 0).any():
            raise RuntimeError("Unexpected negative coordinates in reference mesh points.")
        reor_mesh_points = (reference_mesh_points - reference_mesh_points.min(axis=0)) * scale_factors + reference_mesh_points.min(axis=0)
        reor_deformations = deformations * scale_factors

        return reor_mesh_points, reor_deformations

def align_deformations_to_segmented_volume(
    base_dir: Path,
    dataset_nii_dir: Path,
    dataset_segmentations_dir: Path,
    deformation_index: int,
    patient_index: int,
    lung_side: str = "left",
) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
    volsegloader = VolSegAndDeformLoader(base_dir, dataset_nii_dir, dataset_segmentations_dir)
    (mesh_deformations,
        lung_warp_reference,
        reference_mesh,
        lung_segmentation,
        body_segmentation,
        muscle_fat_segmentation,
        airways_mask
    ) = volsegloader.load_deform_and_volume(
        deformation_index=deformation_index,
        patient_index=patient_index,
        lung=lung_side,
        verbose=True,
        deform_as_proportional_shift=False, # Absolute coordinates
    )

    lsp = LungSegmentationPostProcessor(lung_warp_reference, lung_segmentation, airways_mask)
    lung_segmentation_better, centroids, lung_warp_ref_reoriented, reference_mesh, mesh_deformations = lsp.process(reorient=True, reference_mesh_to_reorient=reference_mesh, deformations_to_reorient=mesh_deformations) # Just reorienting segmentation

    warp_ref_pad_shape = np.array(lung_segmentation_better.shape) - np.array(lung_warp_ref_reoriented.shape)
    # Handle positive differences
    lung_warp_ref_reoriented = np.pad(
        lung_warp_ref_reoriented,
        pad_width=np.stack([np.where(warp_ref_pad_shape > 0, warp_ref_pad_shape, 0), np.zeros(3)]).astype(int).T,
        mode='constant',
        constant_values=0
    )
    # Handle negative differences (crop)
    lung_warp_ref_reoriented = lung_warp_ref_reoriented[tuple(
        slice(max(-d//2, 0), s - max(-(d - d//2), 0))
        for s, d in zip(lung_warp_ref_reoriented.shape, warp_ref_pad_shape)
    )]


    intermediate_warp_ref = lung_warp_ref_reoriented.copy()
    intermediate_mesh_deformations = mesh_deformations.copy()
    intermediate_mesh_points = reference_mesh.points.copy()

    transformer = AffineTransformer(lung_side)

    intermediate_warp_ref = transformer.rotate_warp_ref(intermediate_warp_ref, lung_segmentation_better)
    intermediate_mesh_points, intermediate_mesh_deformations = transformer.rotate_points(intermediate_mesh_points, intermediate_mesh_deformations, lung_segmentation_better)

    intermediate_warp_ref = transformer.scale_warp_ref(intermediate_warp_ref, lung_segmentation_better)
    intermediate_mesh_points, intermediate_mesh_deformations = transformer.scale_points(intermediate_mesh_points, intermediate_mesh_deformations, lung_segmentation_better)

    intermediate_warp_ref = transformer.translate_warp_ref(intermediate_warp_ref, lung_segmentation_better)
    intermediate_mesh_points, intermediate_mesh_deformations = transformer.translate_points(intermediate_mesh_points, intermediate_mesh_deformations, lung_segmentation_better)

    return intermediate_warp_ref, intermediate_mesh_points, intermediate_mesh_deformations, lung_segmentation_better


if __name__ == "__main__":
    parser = ArgumentParser(description="Apply deformation to lung meshes.")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--dataset-nii-dir", type=str, required=True, help="Directory containing the original .nii.gz files for each patient.")
    parser.add_argument("--dataset-segmentations-dir", type=str, required=True, help="Directory containing the segmentation .nii.gz files for each patient.")
    parser.add_argument("--deformation-index", type=int, help="Index of the deformation to use.")
    parser.add_argument("--patient-index", type=int, help="Index of the patient to use.")
    parser.add_argument("--output-dir", type=str, help="Directory to save the output deformation fields and registered meshes. (Relative to --data-dir)")
    args = parser.parse_args()
    data_base_dir = Path(args.data_dir)

    with ProcessPoolExecutor(max_workers=2) as executor:
        future_left = executor.submit(
            align_deformations_to_segmented_volume,
            base_dir=data_base_dir,
            dataset_nii_dir=Path(args.dataset_nii_dir),
            dataset_segmentations_dir=Path(args.dataset_segmentations_dir),
            deformation_index=args.deformation_index,
            patient_index=args.patient_index,
            lung_side="left",
        )
        future_right = executor.submit(
            align_deformations_to_segmented_volume,
            base_dir=data_base_dir,
            dataset_nii_dir=Path(args.dataset_nii_dir),
            dataset_segmentations_dir=Path(args.dataset_segmentations_dir),
            deformation_index=args.deformation_index,
            patient_index=args.patient_index,
            lung_side="right",
        )

        lung_warp_ref_left, ref_mesh_left, rel_def_left, lungs_seg = future_left.result()
        lung_warp_ref_right, ref_mesh_right, rel_def_right, lungs_seg = future_right.result()

    ref_mesh_all = np.concatenate([ref_mesh_left, ref_mesh_right], axis=0)
    rel_def_all = np.concatenate([rel_def_left, rel_def_right], axis=1)

    output_dir = data_base_dir / args.output_dir
    output_dir.mkdir(exist_ok=True)

    np.save(
        output_dir / f"mesh_relative_deformations_patient_{args.patient_index:02d}_deform_{args.deformation_index:02d}.npy",
        rel_def_all
    )
    np.save(
        output_dir / f"registered_mesh_points_patient_{args.patient_index:02d}_deform_{args.deformation_index:02d}.npy",
        ref_mesh_all
    )
    patient_segmentation = output_dir / f"cleaned_lung_segmentation_patient_{args.patient_index:02d}.npy"

    if not patient_segmentation.exists():
        np.save(patient_segmentation, lungs_seg)
    print("Deformation application complete. Results saved to", output_dir, file=stderr)
