#!/usr/bin/env python
# coding: utf-8

import numpy as np
import cupy as cp

from sys import path
from os import cpu_count
from pathlib import Path
from cupyx.scipy.ndimage import binary_opening, binary_dilation
from skimage.morphology import ball, remove_small_objects
from tqdm.auto import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager, Lock
from typing import Union
from argparse import ArgumentParser
from itertools import cycle, product
from traceback import format_exc
from skimage.segmentation import flood_fill
from torch.cuda import device_count
from totalsegmentator.python_api import totalsegmentator

path.append(str(Path('.').resolve()))

from utils import load_nii, save_nii

def normalise_ct(ct_data: np.ndarray) -> np.ndarray:
    ct_data = ct_data.astype(np.float32)
    ct_data -= ct_data[ct_data != -1024].min() # Find min value that's not padding
    return ct_data.astype(np.int16)

def flood_fill_remove_background(input: str, output: str, flood_tolerance: int, as_mask: bool = False) -> None:
    ct_data, ct_vol = load_nii(input, return_nii_obj=True)

    vol = ct_data.copy()
    for corner in list(product(*zip([0,0,0], np.array(ct_data.shape) - 1))):
        vol = flood_fill(vol, corner, new_value=-1024, tolerance=flood_tolerance) # Set unwanted values to background
    
    # Flood fill from corners
    # vol = normalise_ct(vol)

    if as_mask:
        vol = (vol > vol.min() + 100).astype(np.uint8)

    save_nii(
        volume=vol,
        path=output,
        reference_nii_obj=ct_vol,
        save_dtype=ct_data.dtype
    )

def make_inside_lung_mask_basic(body_ct: Path, in_case: Path, output_dir: Path) -> None:
    ct_data, ct_vol = load_nii(body_ct, return_nii_obj=True)

    in_case_base = in_case.name.replace("".join(in_case.suffixes), "")
    ct_data_gpu = cp.asarray(ct_data)
    new_vol = (ct_data_gpu < cp.median(ct_data_gpu)) & (ct_data_gpu > -1024)

    # Keep the connected components with the largest volume. Should be over 10% of the total volume
    new_vol = binary_opening(new_vol, structure=cp.asarray(ball(5)))
    new_vol = remove_small_objects(new_vol.get(), min_size=int(0.1 * np.prod(ct_data.shape)), connectivity=2)

    save_nii(
        volume=new_vol.astype(np.uint8),
        path=output_dir / (in_case_base + '_lung_mask.nii.gz'),
        reference_nii_obj=ct_vol,
        save_dtype=np.uint8
    )

def make_inside_lung_mask(body_ct: Path, in_case: Path, output_dir: Path, device: str) -> None:
    _, ct_vol = load_nii(body_ct, return_nii_obj=True)

    in_case_base = in_case.name.replace("".join(in_case.suffixes), "")
    temp_output_lung_mask = output_dir / (in_case_base + '_temp_lung_mask.nii.gz')

    totalsegmentator(
        input=str(body_ct),
        output=str(temp_output_lung_mask),
        ml=True,
        nr_thr_resamp=cpu_count() // 2,
        nr_thr_saving=cpu_count() // 2,
        device=device,
        task='total',
        roi_subset_robust=["lung_upper_lobe_left", "lung_lower_lobe_left", "lung_upper_lobe_right", "lung_middle_lobe_right", "lung_lower_lobe_right"],
        quiet=True
    )
    temp_output_vol = load_nii(temp_output_lung_mask, return_nii_obj=False)
    lung_mask = temp_output_vol > 0

    temp_output_lung_mask.unlink()

    output_lung_mask = output_dir / (in_case_base + '_lung_mask.nii.gz')
    output_lung_mask.touch()

    save_nii(
        volume=lung_mask,
        path=output_lung_mask,
        reference_nii_obj=ct_vol,
        save_dtype=np.uint8
    )

def mask_out_with_body_seg(ct_data_path: Path, body_seg_path: Path, output_path: Path) -> None:
    ct_data, ct_vol = load_nii(ct_data_path, return_nii_obj=True)
    body_data = load_nii(body_seg_path, return_nii_obj=False)

    # Get body_data segment 1
    body_mask = body_data == 1
    ct_data_body_only = ct_data.copy()
    ct_data_body_only[~body_mask] = -1024

    save_nii(
        volume=ct_data_body_only,
        path=output_path,
        reference_nii_obj=ct_vol,
        save_dtype=ct_data.dtype
    )

def thresholding_muscle_and_fat_seg(in_case_body_only: Path, lung_mask_path: Path, output_dir: Path) -> None:
    ct_data, ct_vol = load_nii(in_case_body_only, return_nii_obj=True)
    
    lung_mask = load_nii(lung_mask_path, return_nii_obj=False).astype(bool)

    blood_mask = (ct_data >= 990) & (ct_data <= 1100)
    muscle_mask = (ct_data >= 800) & (ct_data <= 990)

    blood_mask = blood_mask & (~lung_mask)
    muscle_mask = muscle_mask & (~lung_mask)

    blood_output_path = output_dir / (in_case_body_only.stem + '_blood_mask.nii.gz')
    muscle_output_path = output_dir / (in_case_body_only.stem + '_muscle_mask.nii.gz')

    save_nii(
        volume=blood_mask,
        path=blood_output_path,
        reference_nii_obj=ct_vol,
        save_dtype=np.uint8
    )
    save_nii(
        volume=muscle_mask,
        path=muscle_output_path,
        reference_nii_obj=ct_vol,
        save_dtype=np.uint8
    )

def work_on_case(first_input: Path, output_dir: Path, device: str, pbar: Union[tqdm, None], lock: Lock) -> None:
    if pbar is not None:
        pbar.write(f'Running on: {first_input}\n')
    
    file_base_name = first_input.name.replace("".join(first_input.suffixes), "")

    body_trunc_file = output_dir / (file_base_name + '_body_trunc.nii.gz')
    if not body_trunc_file.exists():
        flood_fill_remove_background( # Initial clean
            input=str(first_input),
            output=str(body_trunc_file),
            flood_tolerance=100,  # For CT, we need to overpower small barriers, but not cross through noisy areas
            as_mask=True # We need a mask for mask_out_with_body_seg
        )
    else:
        print('Skipping', body_trunc_file, 'as it already exists.')

    bod_only_ct_path = output_dir / (file_base_name + '_ct_body_only.nii.gz')
    if not bod_only_ct_path.exists():
        mask_out_with_body_seg(
            ct_data_path=first_input,
            body_seg_path=str(body_trunc_file),
            output_path=bod_only_ct_path
        )
    else:
        print('Skipping', bod_only_ct_path, 'as it already exists.')

    with lock:
        total_output_path = output_dir / (file_base_name + '_total.nii.gz')
        if not total_output_path.exists():
            totalsegmentator(
                input=str(bod_only_ct_path),
                output=str(total_output_path),
                ml=True,
                nr_thr_resamp=cpu_count() // 2,
                nr_thr_saving=cpu_count() // 2,
                device=device,
                task='total',
                roi_subset_robust=['vertebrae_S1', 'vertebrae_L5', 'vertebrae_L4', 'vertebrae_L3', 'vertebrae_L2', 'vertebrae_L1', 'vertebrae_T12', 'vertebrae_T11', 'vertebrae_T10', 'vertebrae_T9', 'vertebrae_T8', 'vertebrae_T7', 'vertebrae_T6', 'vertebrae_T5', 'vertebrae_T4', 'vertebrae_T3', 'vertebrae_T2', 'vertebrae_T1', 'vertebrae_C7', 'vertebrae_C6', 'vertebrae_C5', 'vertebrae_C4', 'vertebrae_C3', 'vertebrae_C2', 'vertebrae_C1', 'heart', 'aorta', 'common_carotid_artery_left', 'common_carotid_artery_right', 'rib_left_1', 'rib_left_2', 'rib_left_3', 'rib_left_4', 'rib_left_5', 'rib_left_6', 'rib_left_7', 'rib_left_8', 'rib_left_9', 'rib_left_10', 'rib_left_11', 'rib_left_12', 'rib_right_1', 'rib_right_2', 'rib_right_3', 'rib_right_4', 'rib_right_5', 'rib_right_6', 'rib_right_7', 'rib_right_8', 'rib_right_9', 'rib_right_10', 'rib_right_11', 'rib_right_12'],
                quiet=True
            )
        else:
            print('Skipping', total_output_path, 'as it already exists.')

        lung_vessels_output_path = output_dir / (file_base_name + '_lung_vessels.nii.gz')
        if not lung_vessels_output_path.exists():
            totalsegmentator(
                input=str(bod_only_ct_path),
                output=str(lung_vessels_output_path),
                ml=True,
                nr_thr_resamp=cpu_count() // 2,
                nr_thr_saving=cpu_count() // 2,
                device=device,
                task='lung_vessels',
                quiet=True
            )
        else:
            print('Skipping', lung_vessels_output_path, 'as it already exists.')

        output_lung_mask = output_dir / (file_base_name + '_lung_mask.nii.gz')
        if not output_lung_mask.exists():
            make_inside_lung_mask_basic(
                bod_only_ct_path,
                lung_vessels_output_path,
                output_dir
            )
        else:
            print('Skipping', output_lung_mask, 'as it already exists.')

        output_muscle_fat_path = output_dir / (file_base_name + '_muscle_fat.nii.gz')
        if not output_muscle_fat_path.exists():
            totalsegmentator(
                input=str(bod_only_ct_path),
                output=str(output_muscle_fat_path),
                ml=True,
                nr_thr_resamp=cpu_count() // 2,
                nr_thr_saving=cpu_count() // 2,
                device=device,
                task='tissue_types_mr', # This requires a special license. Academic users, request here: https://backend.totalsegmentator.com/license-academic/
                quiet=True              # Yes, it makes no sense to use the MR model on CT. But it hasn't learned on absolute HU values, but CT. This is good for generalisability, and it works. 
            )
        else:
            print('Skipping', output_muscle_fat_path, 'as it already exists.')


if __name__ == '__main__':
    parser = ArgumentParser(description="Process CT lung tissue identification.")
    parser.add_argument('--input-dir', type=str, required=True, help="Directory containing input .nii.gz files.")
    parser.add_argument('--output-dir', type=str, required=True, help="Directory to save output segmentations.")
    parser.add_argument('--max-workers', type=int, default=cpu_count() // 12, help="Maximum number of parallel workers.")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    input_files = list(input_dir.glob("*.nii.gz")) + list(input_dir.glob("*.nii"))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # input_files

    # for case in (pbar := tqdm(input_files, desc="Processing cases")):
    #     work_on_case(case, output_dir, 'gpu', pbar)

    manager = Manager()
    lock = manager.Lock()
    with tqdm(total=len(input_files), desc="Processing cases") as pbar:
        with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {executor.submit(work_on_case, case, output_dir, device, None, lock): case for case, device in zip(input_files, cycle([f'gpu:{n}' for n in range(device_count())]))}
            for future in as_completed(futures):
                case = futures[future]
                try:
                    future.result()
                except Exception as e:
                    print(f"Error processing case {case}: {format_exc()}")
                    # Kill all processes and exit
                    executor.shutdown(wait=False, cancel_futures=True)
                    exit(1)
                finally:
                    pbar.update(1)
