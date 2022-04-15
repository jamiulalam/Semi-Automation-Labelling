import os, sys
try:
    from lib.stltovoxel.main import convert_files
except:
    from stltovoxel.main import convert_files
import numpy as np

from vtk.util import numpy_support
import vtk
import numpy as np

import os
import numpy as np
from glob import glob
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


 ## SEMI-AUTOMATEDLABELING

def load_stl_files(root):
    from glob import glob
    import os
    stl_components = glob(os.path.join(root, '*.STL')) 
    number_of_colors = len(stl_components)
    segment_color, color_tuples = generate_random_colors(number_of_colors)
    return segment_color, color_tuples, stl_components

def generate_random_colors(number_of_colors):
    import matplotlib.pyplot as plt
    import random
    colors = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
                for i in range(number_of_colors)]
    from PIL import  ImageColor
    color_tuples = [ImageColor.getcolor(color, "RGB") for color in colors]
    return colors, color_tuples
             
def export_ColorTable(mask_volume, stl_components, color_tuples, ColorTable_file):
    voxels = np.unique(mask_volume)
    print('size stls = ', len(stl_components))
    print('size colors = ', len(color_tuples))
    print('size volume segment  = ', len(voxels))
    print('volume segment  = ', voxels)

    ColorTable = []
    ColorTable.append([0, 'Background', 0, 0, 0, 0])
    stl_components2 = [os.path.basename(path) for path in stl_components]
    idx = 1
    for component, color in  zip(stl_components2, color_tuples) :
        ColorTable.append([voxels[idx], component, color[0], color[1], color[2], 255 ])
        idx = idx + 1

    print(ColorTable)
    textfile = open(ColorTable_file, "w")
    for element in ColorTable:
        for j in element:
            textfile.write(str(j) + ' ')
        textfile.write( "\n")
    textfile.close()

    return ColorTable

def generate_mask_from_stl(root, volume_path='', ColorTable_file='', resolution=1000, parallel=True):
    # load stls 
    colors, color_tuples, stl_components = load_stl_files(root)   
    print( '\n\n The number of found STL components =', len(stl_components ) )
    print( '\n\n STL components =', stl_components )
    # mask volume
    if volume_path == '':
        volume_path = os.path.join(root , 'mask_volume.nrrd')
    else:
        volume_path = os.path.join(volume_path , 'mask_volume.nrrd')
    #color table :
    if ColorTable_file == '':
        ColorTable_file = os.path.join(root , 'mask_volume_ColorTable.ctbl')
    else:
        ColorTable_file = os.path.join(ColorTable_file , 'mask_volume_ColorTable.ctbl')

    ## generate the volume:
    mask_volume = convert_files(stl_components, volume_path, colors=color_tuples, resolution=resolution, parallel=parallel)
    #Export color table :
    ColorTable = export_ColorTable(mask_volume, stl_components, color_tuples, ColorTable_file)
    # done
    print('\n\n The mask and labels are generated successefully !!!!')
    return mask_volume, ColorTable

def load_image(path):
    import numpy as np
    import cv2
    print(' The selected image is :', path)
    filename, file_extension = os.path.splitext(path)
    img = cv2.imread(path,0)
    return img

def napari_view_volume(volume):
    import napari
    napari.gui_qt()
    # viewer = napari.view_image(img3D_pos, colormap='magma')
    # %gui qt
    viewer = napari.Viewer()
    viewer.add_image(volume,name='defect-free scan', colormap='gray')#colormap='gist_earth')
    return viewer



def get_vtk_volume_from_3d_array(data):
    
  # check the vtk required format
  if data.shape[0] != data.shape[2]:
    data = np.transpose(data, (0, 2, 1))

  imdata = vtk.vtkImageData()
  depthArray = numpy_support.numpy_to_vtk(data.ravel())#, deep=True, array_type=vtk.VTK_DOUBLE)
  print('data.shape  = ', data.shape)
  imdata.SetDimensions(data.shape)
  spacing= 1
  imdata.SetSpacing([spacing,spacing,spacing])
  imdata.SetOrigin([0,0,0])
  imdata.GetPointData().SetScalars(depthArray)

  colorFunc = vtk.vtkColorTransferFunction()
  # colorFunc.AddRGBPoint(1, 1, 0.0, 0.0) # Red
  colorFunc.AddRGBPoint(2, 0.0, 1, 0.0) # Green
  opacity = vtk.vtkPiecewiseFunction()
  volumeProperty = vtk.vtkVolumeProperty()
  # volumeProperty.SetColor(colorFunc)
  volumeProperty.SetScalarOpacity(opacity)
  volumeProperty.SetInterpolationTypeToLinear()
  volumeProperty.SetIndependentComponents(2)

  volumeMapper = vtk.vtkOpenGLGPUVolumeRayCastMapper()
  volumeMapper.SetInputData(imdata)
  volumeMapper.SetBlendModeToMaximumIntensity()


  volume = vtk.vtkVolume()
  volume.SetMapper(volumeMapper)
  volume.SetProperty(volumeProperty)
  return volume
