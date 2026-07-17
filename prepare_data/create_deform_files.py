import numpy as np
import h5py
import pyvista as pv
import ants
import trimesh

from pathlib import Path
from tqdm.auto import tqdm
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from scipy.io import loadmat
from pandas import DataFrame
from warnings import warn

from numpy.typing import NDArray
from typing import List, Tuple, Dict, Union

def make_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Create case directory for a patient, with files for the mesh positions over time.")
    parser.add_argument("--masks-file", type=str, required=True, help="Path to the masks file (H5 format OR MAT format).")
    return parser

def mask_to_surface_mesh(mask: NDArray[np.int8], time_index: Union[int, None] = None) -> Tuple[pv.PolyData, pv.PolyData]:
    mask_t_pv = pv.ImageData(
        dimensions=mask.shape,
        spacing=(1.0, 1.0, 1.0),
        origin=(0.0, 0.0, 0.0)
    )
    mask_t_pv.point_data["mask"] = mask.ravel(order='F')

    conn = mask_t_pv.connectivity('all', scalar_range=[1, mask.max()])
    labeled_array = conn.get_array('RegionId')
    num_features = np.max(labeled_array)
    if num_features > 2:
        component_sizes = np.array([np.sum(labeled_array == i) for i in range(0, num_features)])
        selected_components = np.argsort(component_sizes)[-2:]
        warning_message = (f"At time {time_index}: " if time_index is not None else "") + f"Found {num_features} connected components in the mask, of size {component_sizes}. Selecting the 2 largest components."
        warn(warning_message)

        masked_sel = np.isin(labeled_array, selected_components)
        conn = conn.extract_cells(masked_sel)
    cell_centers = conn.cell_centers().points
    lung_pts = [cell_centers[conn.get_array("RegionId") == label] for label in [0, 1]]

    lung_meshes = tuple(
        pv.PolyData(pts).delaunay_3d(alpha=1.1, tol=0, progress_bar=False).extract_surface(algorithm=None).clean() for pts in lung_pts
    )
    return lung_meshes

def coregister_images(source_target_pair: Tuple[NDArray[np.complex128], NDArray[np.complex128]], mask_pair: Tuple[NDArray[np.int8], NDArray[np.int8]]) -> Dict:
    source, target = source_target_pair
    source = ants.from_numpy(np.abs(source))
    target = ants.from_numpy(np.abs(target))
    mask_shape = mask_pair[0].shape
    source_mask = ants.from_numpy(mask_pair[0])
    target_mask = ants.from_numpy(mask_pair[1])

    source = ants.resample_image(source, mask_shape, use_voxels=True)
    target = ants.resample_image(target, mask_shape, use_voxels=True)

    reg = ants.registration(
        fixed=source,
        moving=target,
        mask=source_mask,
        moving_mask=target_mask,
        type_of_transform='ElasticSyN'
    )
    return reg

def warp_mesh(mesh: pv.PolyData, reg: List) -> NDArray[np.float32]:
    d = np.asarray(mesh.points)
    d = DataFrame(d, columns=["x", "y", "z"])

    res = ants.apply_transforms_to_points(dim=3, points=d, transformlist=reg)
    return res.to_numpy()

if __name__ == "__main__":
    args = make_parser().parse_args()

    # Support multiple deformation formats used by the lab, but convert to a common format (T, X, Y, Z) for processing
    if Path(args.masks_file).suffix == '.h5':
        with h5py.File(args.masks_file, "r") as f:
            masks: NDArray[np.int8] = f["mask"][:] # size (T, X, Y, Z)
    elif Path(args.masks_file).suffix == '.mat':
        mat: NDArray[np.int8] = loadmat(args.masks_file)['dynamic_mask'] # size (X, Y, Z, T)
        masks = np.transpose(mat, (3, 0, 1, 2)) # size (T, X, Y, Z)

    print("Data loaded.")

    with ProcessPoolExecutor(max_workers=masks.shape[0]) as executor:
        lung_meshes = list(executor.map(mask_to_surface_mesh, masks, range(masks.shape[0])))
    
    print("Meshes extracted from masks.")
    
    lung_meshes_tri = [(pv.to_trimesh(p[0], triangulate=True), pv.to_trimesh(p[1], triangulate=True)) for p in lung_meshes]

    starting_meshes = lung_meshes_tri[0]

    meshes_over_time = [starting_meshes]
    with ProcessPoolExecutor(max_workers=2) as executor:
        for t in tqdm(range(1, masks.shape[0]), desc="Coregistering meshes"):
            warped_pts_l_future = executor.submit(
                trimesh.registration.nricp_amberg,
                source_mesh=meshes_over_time[-1][0],
                target_geometry=lung_meshes_tri[t][0]
            )
            warped_pts_r_future = executor.submit(
                trimesh.registration.nricp_amberg,
                source_mesh=meshes_over_time[-1][1],
                target_geometry=lung_meshes_tri[t][1]
            )
            # Clone mesh, adjust point location, push back
            new_mesh_l = meshes_over_time[-1][0].copy()
            new_mesh_l.vertices = warped_pts_l_future.result()
            new_mesh_r = meshes_over_time[-1][1].copy()
            new_mesh_r.vertices = warped_pts_r_future.result()
            meshes_over_time.append((new_mesh_l, new_mesh_r))

    aligned_meshes_time = [(np.asarray(x[0].vertices), np.asarray(x[1].vertices)) for x in meshes_over_time]

    lung_left_time = np.stack([x[0] for x in aligned_meshes_time]) # size (T, N, 3)
    lung_right_time = np.stack([x[1] for x in aligned_meshes_time]) # size (T, N, 3)

    # Get this file dir
    output_dir = Path(__file__).parent
    patient_id = Path(args.masks_file).parent.parent.parent.stem

    output_file = output_dir / f"{patient_id}_mesh_LL.npy"
    np.save(output_file, lung_left_time)
    output_file = output_dir / f"{patient_id}_mesh_RL.npy"
    np.save(output_file, lung_right_time)
