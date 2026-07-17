This repository is to help you process raw CT data + segmented dynamic MRI volumes into a format ready for creating simulation phantoms.

It is split into 3 parts: downloading CT data (only if required, recommended to use your own data), CT segmentation, and data preparation for simulators.

First, if required, you may run `download_data.sh`, which will pull a good-quality CT dataset from Zenodo (Lung CT Deformable Image Registration Validation Dataset (Version 1) [Dataset], Criscuolo et al., https://doi.org/10.5281/zenodo.8200423).

Following this, you may use the `segmentation` and `prepare_data` directories. They are supposed to be visited / run in that order. Each of the two directories contains a `requirements.txt` for a virtual environment to complete the task, as well as a dedicated `README.md`. The expected version of Python to use is at the top of the requirements file.

> [!important]
> It is expected for you to run all these files from the top-level directory for imports and relative paths to resolve correctly. (i.e. run `python segmentation/ct_lung_tissue_id.py`, and not `cd segmentation; python ct_lung_tissue_id.py`.)
