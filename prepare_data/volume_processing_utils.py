import ants
import numpy as np

from numpy.typing import NDArray
from typing import List, Tuple, Union
from pathlib import Path
from pyvista import UnstructuredGrid, PolyData
from vedo import Mesh

from scipy.ndimage import shift
from skimage.measure import centroid
from skimage.morphology import remove_small_objects, remove_small_holes

def hull_to_volume(hull: UnstructuredGrid) -> NDArray[np.bool_]:
    """
    Convert a closed hull mesh into a dense boolean volume.
    Voxels are aligned with hull coordinates.
    
    Returns
    -------
    volume : NDArray[np.bool_]
        3D array where True indicates inside the hull.
    """

    surface = hull.delaunay_3d(alpha=1.1, tol=0).extract_surface(algorithm=None).clean()

    surface_vedo = Mesh(surface)
    surface_vedo = surface_vedo.normalize().wireframe()
    volume = surface_vedo.binarize(values=(1,0), dims=(hull.points.max(axis=0) * 1.2).astype(int))
    volume = volume.tonumpy().astype(bool)

    # Only translation required to align with hull coordinates
    surface_centroid = surface.points.mean(axis=0)
    volume_centroid = centroid(volume)

    translation = surface_centroid - volume_centroid
    # Apply translation to volume
    volume = shift(volume.astype(float), shift=translation, order=0).astype(bool)

    return volume

def convert_pat_deform_to_ants(patient_path: Union[str, Path], rescaling_reference: Union[ants.ANTsImage, None] = None, full_volume: bool = True, deform_as_proportional_shift: bool = True) -> Tuple[Union[List[ants.ANTsTransform], NDArray], ants.ANTsImage, ants.ANTsImage, Union[UnstructuredGrid, None]]:
    '''
    Convert the deformation fields stored in this folder.
    If full_volume is False, returns deformations as a shift in mesh vertices, along with mesh, reference grid, and volume at t=0.
    If full_volume is True, returns list of ANTs transforms, reference grid, and volume at t=0.
    If deform_as_proportional_shift is True, the deformation fields are interpreted as proportional shifts (i.e., percentage of original position), wrt the reference mesh.
    '''
    patient_path = Path(patient_path)

    deformations = np.load(patient_path) # size (T, Npts, 3)

    mesh_zero = PolyData(deformations[0].copy())
    volume_zero = hull_to_volume(mesh_zero)
    volume_zero = remove_small_objects(volume_zero.astype(bool), max_size=1000)
    volume_zero = remove_small_holes(volume_zero, max_size=100)
    volume_zero = ants.core.ants_image_io.from_numpy(volume_zero)

    reference_mesh = PolyData(deformations[0].copy())
    if deform_as_proportional_shift:
        deformations = deformations / deformations[0] # Proportional shift

    spacing = 1.0  # grid spacing
    min_coords = np.min(deformations[0], axis=0)
    max_coords = np.max(deformations[0], axis=0)

    # Create a 3D grid covering the mesh
    grid_x, grid_y, grid_z = np.mgrid[
        min_coords[0]:max_coords[0]:spacing,
        min_coords[1]:max_coords[1]:spacing,
        min_coords[2]:max_coords[2]:spacing
    ]

    reference_grid = ants.from_numpy(grid_x, spacing=(spacing, spacing, spacing))

    deformations -= deformations[0]  # Convert to shifts from reference position (actual deformations)

    if not full_volume:
        return deformations, reference_grid, volume_zero, reference_mesh
    else:
        raise RuntimeError("Full volume deformation conversion is no longer supported at this level.")
