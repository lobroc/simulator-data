This folder is to segment raw CT data, and determine what types of tissues lay where in the volume.

This is step 2 / 3 in data generation: download, segmentation, prepare.

Here is a quick run-down of the file to run. Run this one only:
`ct_lung_tissue_id.py` -> Create segmentations of CT data and save into dedicated volumes for each type of segmentation done (blood vessels and airways, tissue types, lung, etc.)

`retrieve_segmentations.py` -> Extra code to help with loading back previous segmentations. Used in other parts of this codebase.

All code in this directory uses the dedicated `total-segmentator-venv`, see the requirements file in this diretory. Uses python 3.14
