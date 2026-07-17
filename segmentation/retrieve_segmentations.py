import ants

from pathlib import Path
from typing import List
from concurrent.futures import ThreadPoolExecutor

def load_segmentations_for_img(segmentations_dir: Path, case_name: str) -> List[ants.ANTsImage]:
    case_name = Path(case_name)
    case_base_name = case_name.name.replace("".join(case_name.suffixes), "")
    segmentation_files = {'general': segmentations_dir / f"{case_base_name}_total.nii.gz",
                     'airways_and_vessels': segmentations_dir / f"{case_base_name}_lung_vessels.nii.gz",
                     'body': segmentations_dir / f"{case_base_name}_body_trunc.nii.gz",
                     'lungs': segmentations_dir / f"{case_base_name}_lung_vessels_lung_mask.nii.gz",
                     'muscle_fat': segmentations_dir / f"{case_base_name}_muscle_fat.nii.gz",
                    #  'low_density': segmentations_dir / f"{case_base_name}_ct_body_only_blood_mask.nii.gz",
                    #  'muscle': segmentations_dir / f"{case_base_name}_ct_body_only_muscle_mask.nii.gz"
                    }

    for key, path in segmentation_files.items():
        assert path.exists(), f"Segmentation file {path} (for {key}) does not exist."

    segmentation_objs = dict()
    with ThreadPoolExecutor(max_workers=len(segmentation_files)) as executor:
        futures = {key: executor.submit(ants.image_read, str(path)) for key, path in segmentation_files.items()}
        for key, future in futures.items():
            segmentation_objs[key] = future.result()

    def change_ids(segmentation: ants.ANTsImage, save_path: Path, id_map: dict) -> ants.ANTsImage:
        '''
        Map duplicate segmentation IDs to different ones. Write to disk and return a copy to new data.
        '''

        seg_np = segmentation.numpy() # this is a copy

        for old_id, new_id in id_map.items():
            seg_np[seg_np == old_id] = new_id

        seg_new = ants.from_numpy(seg_np, origin=segmentation.origin, spacing=segmentation.spacing, direction=segmentation.direction)
        seg_new.image_write(str(save_path))

        return seg_new

    # Keep body as 1, change others. General has separate IDs, 27-115 (https://github.com/wasserth/TotalSegmentator#class-details)
    with ThreadPoolExecutor(max_workers=len(segmentation_files)) as executor:
        futures = dict()
        # futures['low_density'] = executor.submit(change_ids, segmentation_objs['low_density'], segmentation_files['low_density'], {1: 2})
        futures['lungs'] = executor.submit(change_ids, segmentation_objs['lungs'], segmentation_files['lungs'], {1: 3})
        # futures['muscle'] = executor.submit(change_ids, segmentation_objs['muscle'], segmentation_files['muscle'], {1: 4})
        futures['airways_and_vessels'] = executor.submit(change_ids, segmentation_objs['airways_and_vessels'], segmentation_files['airways_and_vessels'], {1: 5, 2: 6})

        for key in futures.keys():
            segmentation_objs[key] = futures[key].result()

    return segmentation_objs
