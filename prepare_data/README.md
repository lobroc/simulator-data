This folder is to prepare segmented CT data and MRI data to be fed into the simulator of your choice.
This folder contains code to perform segmentations on CT data, but dynamic MRI should already be segmented by some other method.

This is step 3 / 3 in data generation: download, segmentation, prepare.

Here is a quick run-down of the files and what they do. Run them in this order:
1. `create_deform_files.py` -> Create moving meshes from segmented 4D lung MRI volumes.
2. `align_deformations_to_volume.py` -> Perform affine transformations to align deformation fields onto segmented CT volumes.
> [!important]
> Your source segmentations and CT images might have a "non-standard" orientation with respect to those utilised in this project.
> If you see an incorrect orientation in your alignments, modify the `rotate_warp_ref` and `rotate_points` methods within the `AffineTransformer` class, in `align_deformations_to_volume.py`. Commented examples exist in those functions.

3. `prepare_data_for_simulator.py` -> Assign properties in space, based on segmentations. Put these volumes and deformations together into dedicated patient / deformation directory, for future use in simulations module.

`volume_processing_utils.py` -> Extra code to help with deformations and alignment.

All code in this directory uses the dedicated `mesh-warp-venv`, see the requirements file in this diretory. Uses python 3.13
