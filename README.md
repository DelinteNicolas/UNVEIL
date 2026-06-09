# UNVEIL
[![PyPI](https://img.shields.io/pypi/v/unveil-python?label=pypi%20package)](https://pypi.org/project/unveil-python/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/unveil-python)](https://pypi.org/project/unveil-python/)
![GitHub repo size](https://img.shields.io/github/repo-size/DelinteNicolas/unveil)


UNVEIL is a Python-based visualization tool for interactive exploration of diffusion MRI tractography, anatomical volumes, cortical surfaces, and ROI segmentations. It combines a 3D PyVista viewer with an orthogonal slice viewer.

<img width="1297" height="757" alt="image" src="https://github.com/user-attachments/assets/28807e18-17fb-485b-a8ce-2179cdef796b" />

## Install

The GUI is launched with the following python command in a python console:

```
import unveil.__main__
```
or with
```
python -c "import unveil.__main__"
```
from a terminal with a python environment activated.

## Features

- **3D Tractography Visualization**
    - Load and display `.trk` streamline files.
    - RGB orientation coloring or scalar-based coloring from NIfTI volumes.
    - Adjustable opacity and point/line rendering.
- **Volume Rendering**
    - Load `.nii.gz` images.
    - Interactive volume rendering and orthogonal slice planes.
    - Adjustable opacity.
- **ROI Visualization**
    - Load ROI masks from `.nii` / `.nii.gz`.
    - Automatic surface extraction and smoothing.
    - Individual visibility and color control.
- **Surface Visualization**
    - Load GIFTI (`.gii`) cortical surfaces.
    - Adjustable opacity and smooth shading.
- **Orthogonal Viewer**
    - Axial, coronal, and sagittal views.
    - Mouse-wheel slice navigation.
    - Click-to-navigate crosshair synchronization.
    - ROI overlay display.
    - Screenshot export.
- **Scene Management**
    - Actor visibility control.
    - ROI color editing.
    - Background color toggle.
    - Color-blind-friendly tractography coloring.
- **Export**
    - 3D screenshots.
    - Orthogonal-view screenshots.
    - Animated 360° GIF generation.
