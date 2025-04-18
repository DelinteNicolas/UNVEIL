# -*- coding: utf-8 -*-
"""
Created on Tue Mar  4 13:03:28 2025

@author: nicol
"""

import sys
import numpy as np
import nibabel as nib
import pyvista as pv
from pyvistaqt import QtInteractor  # For embedding PyVista in PyQt6
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (QApplication, QWidget, QHBoxLayout, QVBoxLayout,
                             QPushButton, QFileDialog, QCheckBox, QSlider,
                             QLabel, QComboBox, QMainWindow)
from dipy.io.streamline import load_tractogram
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from unravel.utils import get_streamline_density
# from unravel.viz import plot_trk   # We'll use our local version below


def plot_trk(trk_file, scalar=None, color_map='plasma', opacity: float = 1,
             show_points: bool = False, background: str = 'black', plotter=None,
             name=None):
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
                   cmap=color_map, clim=color_lim, scalars=scalars, name=name)

    elif color_map == 'flesh':
        p.add_mesh(mesh, opacity=opacity, diffuse=0.4, ambient=ambient,
                   interpolate_before_map=False, render_lines_as_tubes=True,
                   line_width=2, point_size=point_size, rgb=rgb,
                   color=[250, 225, 210], name=name)
    else:
        p.add_mesh(mesh, opacity=opacity, diffuse=diffuse, ambient=ambient,
                   interpolate_before_map=False, render_lines_as_tubes=True,
                   line_width=2, point_size=point_size, rgb=rgb,
                   cmap=color_map, scalars=scalars, name=name)

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
        # Create the main horizontal layout
        main_layout = QHBoxLayout(self)

        # Left side: Control panel
        control_layout = QVBoxLayout()
        # self.loadButton = QPushButton('Load .trk File')
        # self.loadButton.clicked.connect(self.loadTrkFile)
        # control_layout.addWidget(self.loadButton)

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
        self.colorMapComboBox.addItems(['plasma', 'flesh'])
        self.colorMapComboBox.currentIndexChanged.connect(
            self.update_trk_viewer)
        control_layout.addWidget(self.colorMapComboBox)

        self.showVolumeCheckbox = QCheckBox('Show Volume')
        self.showVolumeCheckbox.stateChanged.connect(self.update_nii_viewer)
        control_layout.addWidget(self.showVolumeCheckbox)

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

        # self.viewButton = QPushButton('View 3D Plot')
        # self.viewButton.clicked.connect(self.viewPlot)
        # control_layout.addWidget(self.viewButton)

        # Add the control panel layout to the main layout
        main_layout.addLayout(control_layout)

        # Right side: PyVista viewer embedded in the GUI using QtInteractor
        self.plotter = QtInteractor(self)
        main_layout.addWidget(self.plotter.interactor)

        self.setLayout(main_layout)
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

        self.update_trk_viewer()

    def loadNiftiFile(self):
        options = QFileDialog.Options()
        filePath, _ = QFileDialog.getOpenFileName(
            self, "Open .nii.gz File", "", "Nifti Files (*.nii.gz)", options=options)
        if filePath:
            self.nii_file = filePath
            print(f"Loaded NIfTI file: {self.nii_file}")
            self.nii_data = nib.load(filePath).get_fdata()
            grid = pv.ImageData()
            grid.dimensions = np.array(self.nii_data.shape) + 1
            grid.cell_data['values'] = self.nii_data.flatten(order='F')
            self.grid = grid

        self.XSlider.setMaximum(self.nii_data.shape[0])
        self.YSlider.setMaximum(self.nii_data.shape[1])
        self.ZSlider.setMaximum(self.nii_data.shape[2])

        self.update_nii_viewer()

    def set_background_color(self):

        if self.background == 'white':
            self.background = 'black'
        else:
            self.background = 'white'

        self.plotter.background_color = self.background

    def update_trk_viewer(self):

        opacity = self.opacitySlider.value() / 100.0
        show_points = self.showPointsCheckbox.isChecked()
        color_map = self.colorMapComboBox.currentText()

        for file in self.names:

            plot_trk(file, opacity=opacity, plotter=self.plotter,
                     show_points=show_points, color_map=color_map,
                     name=file, background=self.background)

    def _update_nii_x(self):

        if self.showSlicesCheckbox.isChecked():

            center = (self.XSlider.value(),
                      self.YSlider.value(), self.ZSlider.value())

            if self.showXCheckbox.isChecked():
                slice_x = self.grid.slice('x', center)
                self.plotter.add_mesh(slice_x, cmap='grey', name='nii_x',
                                      show_scalar_bar=False, point_size=0,
                                      render_lines_as_tubes=True)
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
                                      render_lines_as_tubes=True)
            else:
                self.plotter.remove_actor('nii_y')

    def _update_nii_z(self):

        if self.showSlicesCheckbox.isChecked():

            center = (self.XSlider.value(),
                      self.YSlider.value(), self.ZSlider.value())

            if self.showZCheckbox.isChecked():
                z_val = self.ZSlider.value()
                slice_z = self.grid.slice('z', center)
                self.plotter.add_mesh(slice_z, cmap='grey', name='nii_z',
                                      show_scalar_bar=False, point_size=0,
                                      render_lines_as_tubes=True)
            else:
                self.plotter.remove_actor('nii_z')

    def update_nii_viewer(self):

        if self.showSlicesCheckbox.isChecked():

            center = (self.XSlider.value(),
                      self.YSlider.value(), self.ZSlider.value())

            self._update_nii_x()
            self._update_nii_y()
            self._update_nii_z()

        else:
            self.plotter.remove_actor('nii_x')
            self.plotter.remove_actor('nii_y')
            self.plotter.remove_actor('nii_z')

        if self.showVolumeCheckbox.isChecked():

            self.plotter.add_volume(self.grid, cmap='gray', opacity=[0.0, 0.045],
                                    show_scalar_bar=False, name='nii_volume')
        else:
            self.plotter.remove_actor('nii_volume')

    def viewPlot(self):

        # Clear previous meshes if necessary
        self.plotter.clear()

        if self.nii_data is not None and self.grid is not None:
            # Add a slice of the NIfTI volume to the scene
            # self.plotter.add_mesh_slice(self.grid, cmap='gray',
            #                             show_scalar_bar=False)

            # self.plotter.add_mesh_slice_orthogonal(self.grid, cmap='gray',
            #                                        show_scalar_bar=False,
            #                                        tubing=False)

            self.update_nii_viewer()

        if self.trk_file is not None:

            self.update_trk_viewer()

        self.plotter.reset_camera()
        self.plotter.render()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.viewer = TrkViewer()
        self.setCentralWidget(self.viewer)
        self.initMenuBar()
        self.setWindowTitle("Tractography Viewer with Menubar")

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
