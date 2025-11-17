# -*- coding: utf-8 -*-
"""
Created on Tue Mar  4 13:03:28 2025

@author: DELINTE Nicolas
"""
import sys
import numpy as np
import nibabel as nib
import pyvista as pv
from pyvistaqt import QtInteractor
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (QApplication, QWidget, QHBoxLayout, QVBoxLayout,
                             QPushButton, QFileDialog, QCheckBox, QSlider,
                             QLabel, QComboBox, QMainWindow, QLineEdit)
from dipy.io.streamline import load_tractogram
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
# from unravel.utils import get_streamline_density
# from unravel.viz import plot_trk   # We'll use our local version below


def gifti_to_pyvista(gii_path):
    """
    Load a GIFTI surface and transform coordinates into voxel space
    using the inverse of the given NIfTI affine.
    """

    # Load GIFTI file
    img = nib.load(gii_path)
    coords_list = img.get_arrays_from_intent('NIFTI_INTENT_POINTSET')
    face_list = img.get_arrays_from_intent('NIFTI_INTENT_TRIANGLE')

    if len(coords_list) == 0 or len(face_list) == 0:
        raise RuntimeError(
            "Could not find POINTSET or TRIANGLE arrays in GIFTI file.")

    # Extract data
    coords_world = coords_list[0].data.astype(np.float32)
    faces = face_list[0].data.astype(np.int32)

    # Build PyVista face array (VTK expects [3, i1, i2, i3, 3, ...])
    faces_pv = np.hstack(
        (np.full((faces.shape[0], 1), 3, dtype=np.int32), faces)).astype(np.int32)

    # Create PolyData mesh in voxel space
    mesh = pv.PolyData(coords_world, faces_pv)

    return mesh


def plot_trk(trk_file, scalar=None, color_map='plasma', opacity: float = 1,
             show_points: bool = False, background: str = 'black', plotter=None,
             name=None, reset_camera=False):
    '''
    3D render for .trk files.

    Parameters
    ----------
    trk_file : str
        Path to tractography file (.trk)
    scalar : 3D array of size (x,y,z), optional
        Volume with values to be projected onto the streamlines.
        The default is None.
    opacity : float, optional
        DESCRIPTION. The default is 1.
    show_points : bool, optional
        Enable to show points instead of lines. The default is False.
    color_map : str, optional
        Color map for the labels or scalar. 'Set3' or 'tab20' recommend for
        segmented color maps. If set to 'flesh', the streamlines are colored
        uniformely with a flesh color. The default is 'plasma'.
    background : str, optional
        Color of the background. The default is 'black'.
    plotter : pyvista.plotter, optional
        If not specifed, creates a new figure. The default is None.

    Returns
    -------
    None.

    '''

    trk = load_tractogram(trk_file, 'same')
    trk.to_vox()
    trk.to_corner()
    streamlines = trk.streamlines

    coord = np.floor(streamlines._data).astype(int)

    l1 = np.ones(len(coord))*2
    l2 = np.linspace(0, len(coord)-1, len(coord))
    l3 = np.linspace(1, len(coord), len(coord))

    lines = np.stack((l1, l2, l3), axis=-1).astype(int)
    lines[streamlines._offsets-1] = 0

    mesh = pv.PolyData(streamlines._data)

    if not show_points:
        mesh.lines = lines
        point_size = 0
        ambient = 0.6
        diffuse = 0.5
    else:
        point_size = 2
        ambient = 0
        diffuse = 1

    if color_map == 'flesh':
        rgb = False
    elif scalar is None:
        point = streamlines._data
        next_point = np.roll(point, -1, axis=0)
        vs = next_point-point
        norm = np.linalg.norm(vs, axis=1)
        norm = np.stack((norm,)*3, axis=1, dtype=np.float32)
        norm = np.divide(vs, norm, dtype=np.float64)
        ends = (streamlines._offsets+streamlines._lengths-1)
        norm[ends, :] = norm[ends-1, :]
        scalars = np.abs(norm)
        rgb = True
    else:
        scalars = scalar[coord[:, 0], coord[:, 1], coord[:, 2]]
        rgb = False

    if plotter is None:
        p = pv.Plotter()
    else:
        p = plotter

    if 'tab' in color_map or 'Set' in color_map:

        N = np.max(scalar)
        cmaplist = getattr(plt.cm, color_map).colors
        cmaplistext = cmaplist*np.ceil(N/len(cmaplist)).astype(int)
        color_map = LinearSegmentedColormap.from_list('Custom cmap',
                                                      cmaplistext[:N], N)
        color_lim = [1, N]

        p.add_mesh(mesh, ambient=ambient, opacity=opacity, diffuse=diffuse,
                   interpolate_before_map=False, render_lines_as_tubes=True,
                   line_width=2, point_size=point_size, rgb=rgb,
                   cmap=color_map, clim=color_lim, scalars=scalars, name=name,
                   reset_camera=reset_camera,
                   user_matrix=trk.affine)

    elif color_map == 'flesh':
        p.add_mesh(mesh, opacity=opacity, diffuse=0.4, ambient=ambient,
                   interpolate_before_map=False, render_lines_as_tubes=True,
                   line_width=2, point_size=point_size, rgb=rgb,
                   color=[250, 225, 210], name=name,
                   reset_camera=reset_camera,
                   user_matrix=trk.affine)
    else:
        p.add_mesh(mesh, opacity=opacity, diffuse=diffuse, ambient=ambient,
                   interpolate_before_map=False, render_lines_as_tubes=True,
                   line_width=2, point_size=point_size, rgb=rgb,
                   cmap=color_map, scalars=scalars, name=name,
                   reset_camera=reset_camera,
                   user_matrix=trk.affine)

    p.background_color = background
    # Do not call p.show() here when using an embedded interactor


class TrkViewer(QWidget):
    def __init__(self):
        super().__init__()
        self.trk_file = None
        self.nii_file = None
        self.nii_data = None
        self.grid = None
        self.names = []
        self.selected_name = None
        self.x = 0
        self.y = 0
        self.z = 0
        self.initUI()
        self.background = 'white'

    def initUI(self):
        # Create the main layout
        global_layout = QVBoxLayout(self)

        # Camera toolbar (buttons for orientation)
        camera_label = QLabel('Camera: ')
        cam_layout = QHBoxLayout()
        btn_iso = QPushButton("Isometric")
        btn_xy = QPushButton("Axe 1")
        btn_xz = QPushButton("Axe 2")
        btn_yz = QPushButton("Axe 3")
        btn_iso.clicked.connect(self.view_isometric)
        btn_xy.clicked.connect(self.view_xy)
        btn_xz.clicked.connect(self.view_xz)
        btn_yz.clicked.connect(self.view_yz)
        cam_layout.addWidget(camera_label)
        cam_layout.addWidget(btn_iso)
        cam_layout.addWidget(btn_xy)
        cam_layout.addWidget(btn_xz)
        cam_layout.addWidget(btn_yz)
        btn_screenshot = QPushButton("Screenshot")
        btn_screenshot.clicked.connect(self.take_screenshot)
        cam_layout.addWidget(btn_screenshot)
        btn_gif = QPushButton("Create GIF")
        btn_gif.setToolTip("This takes ~2min to save after animation end.")
        btn_gif.clicked.connect(self.create_gif)
        cam_layout.addWidget(btn_gif)

        global_layout.addLayout(cam_layout)

        main_layout = QHBoxLayout()
        global_layout.addLayout(main_layout)

        # Left side: Control panel
        control_widget = QWidget()
        control_layout = QVBoxLayout(control_widget)
        control_widget.setMaximumWidth(250)

        self.opacityLabel = QLabel('Opacity:')
        control_layout.addWidget(self.opacityLabel)

        self.opacitySlider = QSlider(Qt.Orientation.Horizontal, self)
        self.opacitySlider.setMinimum(1)
        self.opacitySlider.setMaximum(100)
        self.opacitySlider.setValue(100)
        self.opacitySlider.sliderReleased.connect(self.update_trk_viewer)
        control_layout.addWidget(self.opacitySlider)

        self.showPointsCheckbox = QCheckBox('Show Points')
        self.showPointsCheckbox.stateChanged.connect(self.update_trk_viewer)
        control_layout.addWidget(self.showPointsCheckbox)

        self.colorMapLabel = QLabel('Color Map:')
        control_layout.addWidget(self.colorMapLabel)

        self.colorMapComboBox = QComboBox()
        self.colorMapComboBox.addItems(['rgb', 'flesh', 'scalar'])
        self.colorMapComboBox.currentIndexChanged.connect(
            self.update_trk_viewer)
        control_layout.addWidget(self.colorMapComboBox)
        self.colorMapEdit = QLineEdit()
        self.colorMapEdit.textChanged.connect(self.update_trk_viewer)
        self.colorMapEdit.setToolTip(
            'Insert color map name. Name must be in the matplotlib library. Scalar nii.gz must be loaded.')
        control_layout.addWidget(self.colorMapEdit)

        self.showVolumeCheckbox = QCheckBox('Show Volume')
        self.showVolumeCheckbox.stateChanged.connect(self.update_nii_viewer)
        control_layout.addWidget(self.showVolumeCheckbox)

        self.nii_opacitySlider = QSlider(Qt.Orientation.Horizontal, self)
        self.nii_opacitySlider.setMinimum(0)
        self.nii_opacitySlider.setMaximum(1000)
        self.nii_opacitySlider.setValue(45)
        self.nii_opacitySlider.sliderReleased.connect(self.update_nii_viewer)
        control_layout.addWidget(self.nii_opacitySlider)

        self.showSlicesCheckbox = QCheckBox('Show Slices')
        self.showSlicesCheckbox.stateChanged.connect(self.update_nii_viewer)
        control_layout.addWidget(self.showSlicesCheckbox)

        self.showXCheckbox = QCheckBox('X')
        self.showXCheckbox.stateChanged.connect(self._update_nii_x)
        control_layout.addWidget(self.showXCheckbox)
        self.XSlider = QSlider(Qt.Orientation.Horizontal, self)
        self.XSlider.setMinimum(0)
        self.XSlider.setMaximum(100)
        self.XSlider.setValue(100)
        self.XSlider.sliderReleased.connect(self._update_nii_x)
        control_layout.addWidget(self.XSlider)
        self.showYCheckbox = QCheckBox('Y')
        self.showYCheckbox.stateChanged.connect(self._update_nii_y)
        control_layout.addWidget(self.showYCheckbox)
        self.YSlider = QSlider(Qt.Orientation.Horizontal, self)
        self.YSlider.setMinimum(0)
        self.YSlider.setMaximum(100)
        self.YSlider.setValue(100)
        self.YSlider.sliderReleased.connect(self._update_nii_y)
        control_layout.addWidget(self.YSlider)
        self.showZCheckbox = QCheckBox('Z')
        self.showZCheckbox.stateChanged.connect(self._update_nii_z)
        control_layout.addWidget(self.showZCheckbox)
        self.ZSlider = QSlider(Qt.Orientation.Horizontal, self)
        self.ZSlider.setMinimum(0)
        self.ZSlider.setMaximum(100)
        self.ZSlider.setValue(100)
        self.ZSlider.sliderReleased.connect(self._update_nii_z)
        control_layout.addWidget(self.ZSlider)

        # Gifti
        self.showSurfaceCheckbox = QCheckBox('Show Surface')
        self.showSurfaceCheckbox.stateChanged.connect(self.update_gii_viewer)
        control_layout.addWidget(self.showSurfaceCheckbox)

        self.gii_opacitySlider = QSlider(Qt.Orientation.Horizontal, self)
        self.gii_opacitySlider.setMinimum(0)
        self.gii_opacitySlider.setMaximum(100)
        self.gii_opacitySlider.setValue(15)
        self.gii_opacitySlider.sliderReleased.connect(self.update_gii_viewer)
        control_layout.addWidget(self.gii_opacitySlider)

        # Add the control panel layout to the main layout
        main_layout.addWidget(control_widget)

        # Right side: PyVista viewer embedded in the GUI using QtInteractor
        self.plotter = QtInteractor(self)
        main_layout.addWidget(self.plotter.interactor)

        self.setLayout(global_layout)
        self.setWindowTitle("Tractography Viewer")

    def loadTrkFile(self):
        options = QFileDialog.Options()
        filePath, _ = QFileDialog.getOpenFileName(
            self, "Open TRK File", "", "Tractography Files (*.trk)", options=options)
        if filePath:
            self.trk_file = filePath
            print(f"Loaded TRK file: {self.trk_file}")
            self.names.append(self.trk_file)
            self.selected_name = self.trk_file

        self.update_trk_viewer(reset_camera=True)

    def loadNiftiFile(self):
        options = QFileDialog.Options()
        filePath, _ = QFileDialog.getOpenFileName(
            self, "Open .nii.gz File", "", "Nifti Files (*.nii.gz)", options=options)
        if filePath:
            self.nii_file = filePath
            print(f"Loaded NIfTI file: {self.nii_file}")
            img = nib.load(filePath)
            self.nii_affine = img.affine
            self.nii_data = img.get_fdata()
            grid = pv.ImageData()
            grid.dimensions = np.array(self.nii_data.shape) + 1
            # World coordinates
            # grid.origin = img.affine[:3, 3]
            # grid.spacing = np.diag(img.affine)[:3]
            grid.cell_data['values'] = self.nii_data.flatten(order='F')
            self.grid = grid

        self.XSlider.setMaximum(self.nii_data.shape[0])
        self.YSlider.setMaximum(self.nii_data.shape[1])
        self.ZSlider.setMaximum(self.nii_data.shape[2])

        self.update_nii_viewer(reset_camera=False)

    def loadGiftiFile(self):
        options = QFileDialog.Options()
        filePath, _ = QFileDialog.getOpenFileName(
            self, "Open .gii File", "", "Gifti Files (*.gii)", options=options)
        if filePath:
            self.gii_file = filePath
            print(f"Loaded GIfTI file: {self.gii_file}")
            self.gii_mesh = gifti_to_pyvista(filePath)

        self.update_gii_viewer(reset_camera=False)

    def set_background_color(self):

        if self.background == 'white':
            self.background = 'black'
        else:
            self.background = 'white'

        self.plotter.background_color = self.background

    def view_isometric(self):
        # equivalent to BackgroundPlotter 'iso' camera position
        self.plotter.view_isometric()
        self.plotter.render()

    def view_xy(self):
        # view from +Z axis, XY plane (axial)
        self.plotter.view_xy()   # pyvista method
        self.plotter.render()

    def view_xz(self):
        # view from +Y axis, XZ plane (sagittal)
        self.plotter.view_xz()
        self.plotter.render()

    def view_yz(self):
        # view from +X axis, YZ plane (coronal)
        self.plotter.view_yz()
        self.plotter.render()

    def take_screenshot(self):
        """Take a screenshot of the current 3D view."""
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Screenshot", "",
                                                   "PNG Image (*.png);;JPEG Image (*.jpg);;TIFF Image (*.tif)",
                                                   options=options)
        if file_path:
            self.plotter.screenshot(file_path, transparent_background=True)

    def create_gif(self):
        """Create a 360° rotation GIF of the current 3D view."""
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(self, "Save GIF", "",
                                                   "GIF Files (*.gif)",
                                                   options=options)

        if not file_path:
            return

        # Ensure the file has a .gif extension
        if not file_path.lower().endswith(".gif"):
            file_path += ".gif"

        # Use an offscreen plotter so the main one remains visible
        offscreen = pv.Plotter(
            off_screen=True, window_size=self.plotter.window_size)

        # Copy all meshes and volumes from the main plotter to the offscreen one
        for actor_name, actor in self.plotter.actors.items():
            offscreen.add_actor(actor.copy())

        # Match background color and camera
        offscreen.background_color = self.plotter.background_color
        offscreen.camera_position = self.plotter.camera_position
        offscreen.camera.SetPosition(self.plotter.camera.GetPosition())
        offscreen.camera.SetFocalPoint(self.plotter.camera.GetFocalPoint())
        offscreen.camera.SetViewUp(self.plotter.camera.GetViewUp())

        # Create the 360° rotation GIF
        offscreen.open_gif(file_path, fps=20)
        n_frames = 360
        for i in range(n_frames):
            offscreen.camera.Azimuth(360 / n_frames)
            offscreen.render()
            offscreen.write_frame()
            self.plotter.camera.azimuth += 360 / n_frames
            self.plotter.render()
        offscreen.close()

    def update_trk_viewer(self, reset_camera: bool = False):

        opacity = self.opacitySlider.value() / 100.0
        show_points = self.showPointsCheckbox.isChecked()

        color_map = self.colorMapComboBox.currentText()
        if color_map == 'flesh':
            color_map = 'flesh'
            scalar = None
        elif color_map == 'rgb':
            color_map = 'plasma'
            scalar = None
        else:
            color_map = self.colorMapEdit.text()
            scalar = self.nii_data

        for file in self.names:

            plot_trk(file, opacity=opacity, plotter=self.plotter, scalar=scalar,
                     show_points=show_points, color_map=color_map,
                     name=file, background=self.background,
                     reset_camera=reset_camera)

    def _update_nii_x(self):

        if self.showSlicesCheckbox.isChecked():

            center = (self.XSlider.value(),
                      self.YSlider.value(), self.ZSlider.value())

            if self.showXCheckbox.isChecked():
                slice_x = self.grid.slice('x', center)
                self.plotter.add_mesh(slice_x, cmap='grey', name='nii_x',
                                      show_scalar_bar=False, point_size=0,
                                      render_lines_as_tubes=True,
                                      reset_camera=False,
                                      user_matrix=self.nii_affine)
            else:
                self.plotter.remove_actor('nii_x')

    def _update_nii_y(self):

        if self.showSlicesCheckbox.isChecked():

            center = (self.XSlider.value(),
                      self.YSlider.value(), self.ZSlider.value())

            if self.showYCheckbox.isChecked():
                slice_y = self.grid.slice('y', center)
                self.plotter.add_mesh(slice_y, cmap='grey', name='nii_y',
                                      show_scalar_bar=False, point_size=0,
                                      render_lines_as_tubes=True,
                                      reset_camera=False,
                                      user_matrix=self.nii_affine)
            else:
                self.plotter.remove_actor('nii_y')

    def _update_nii_z(self):

        if self.showSlicesCheckbox.isChecked():

            center = (self.XSlider.value(),
                      self.YSlider.value(), self.ZSlider.value())

            if self.showZCheckbox.isChecked():
                slice_z = self.grid.slice('z', center)
                self.plotter.add_mesh(slice_z, cmap='grey', name='nii_z',
                                      show_scalar_bar=False, point_size=0,
                                      render_lines_as_tubes=True,
                                      reset_camera=False,
                                      user_matrix=self.nii_affine)
            else:
                self.plotter.remove_actor('nii_z')

    def update_nii_viewer(self, reset_camera: bool = False):

        if self.showSlicesCheckbox.isChecked():

            self._update_nii_x()
            self._update_nii_y()
            self._update_nii_z()

        else:
            self.plotter.remove_actor('nii_x')
            self.plotter.remove_actor('nii_y')
            self.plotter.remove_actor('nii_z')

        if self.showVolumeCheckbox.isChecked():
            opacity = self.nii_opacitySlider.value()/1000
            self.plotter.add_volume(self.grid, cmap='gray', opacity=[0, opacity],
                                    show_scalar_bar=False, name='nii_volume',
                                    reset_camera=reset_camera,
                                    user_matrix=self.nii_affine)
        else:
            self.plotter.remove_actor('nii_volume', reset_camera=reset_camera)

    def update_gii_viewer(self, reset_camera: bool = False):

        if self.showSurfaceCheckbox.isChecked():

            opacity = self.gii_opacitySlider.value()/100
            self.plotter.add_mesh(self.gii_mesh, color="ghostwhite",
                                  culling='back', smooth_shading=True,
                                  opacity=opacity, name='gii_surface',
                                  reset_camera=reset_camera,
                                  render_lines_as_tubes=True)
        else:
            self.plotter.remove_actor('gii_surface', reset_camera=reset_camera)
            # self.plotter.actors['gii_surface'].visibility = False


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.viewer = TrkViewer()
        self.setCentralWidget(self.viewer)
        self.initMenuBar()
        self.setWindowTitle("UNVEIL - Tractography Viewer")

    def initMenuBar(self):
        menubar = self.menuBar()

        # File menu
        fileMenu = menubar.addMenu('File')

        # Action to load .trk file
        loadTrkAction = QAction('Load .trk File', self)
        loadTrkAction.triggered.connect(self.viewer.loadTrkFile)
        fileMenu.addAction(loadTrkAction)

        # Action to load .nii.gz file
        loadNiftiAction = QAction('Load .nii.gz File', self)
        loadNiftiAction.triggered.connect(self.viewer.loadNiftiFile)
        fileMenu.addAction(loadNiftiAction)

        # Action to load .gii file
        loadGiftiAction = QAction('Load .gii File', self)
        loadGiftiAction.triggered.connect(self.viewer.loadGiftiFile)
        fileMenu.addAction(loadGiftiAction)

        # Optional: Exit action
        exitAction = QAction('Exit', self)
        exitAction.triggered.connect(self.close)
        fileMenu.addAction(exitAction)

        # View menu
        fileMenu = menubar.addMenu('View')
        setBackColorAction = QAction('Background Color', self)
        setBackColorAction.triggered.connect(self.viewer.set_background_color)
        fileMenu.addAction(setBackColorAction)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
