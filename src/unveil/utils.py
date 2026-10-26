# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 10:03:52 2026

@author: N. Delinte
"""

import numpy as np
from scipy.ndimage import distance_transform_edt


def dilate_atlas_labels(atlas, brain_mask=None, dilation_width: float = 1,
                        min_label: int = 0):
    """
    Dilates cortical atlas labels to include nearby unlabeled voxels (0),
    constrained by a brain mask.
    Copied from 'pilab-binama' package.

    Parameters
    ----------
    atlas : np.ndarray (3D, int)
        Cortical atlas with integer labels (0 = unlabeled).
    brain_mask : np.ndarray (3D, bool or int), optional
        Binary mask of the brain (same shape as atlas). The default is one
    dilation_width : float, optional
        Maximum dilation distance in voxels. The default is None.
    min_label : int, optional
        Labels below this values are not dilated

    Returns
    -------
    dilated_atlas : np.ndarray (3D, int)
        Atlas with dilated labels.
    """

    # Mask unlabeled voxels inside the brain
    unlabeled = np.where(atlas <= min_label, 1, 0)
    if brain_mask is not None:
        unlabeled *= brain_mask.astype('int32')

    # Compute distance transform from labeled voxels
    # Also retrieve indices of nearest labeled voxel for each position
    distances, nearest_idx = distance_transform_edt(unlabeled,
                                                    return_indices=True)

    # Copy atlas to output
    dilated_atlas = atlas.copy()

    # For unlabeled voxels within dilation_width, assign nearest label
    within_dilation = unlabeled & (distances <= dilation_width)

    # Map nearest labeled voxel indices back to atlas labels
    coords = np.argwhere(within_dilation)
    for x, y, z in coords:
        nx, ny, nz = nearest_idx[:, x, y, z]
        dilated_atlas[x, y, z] = atlas[nx, ny, nz]

    return dilated_atlas
