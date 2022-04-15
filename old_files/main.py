import os, sys
import os
import numpy as np
from lib.utils import napari_view_volume, generate_mask_from_stl


def main():
    # STLs root folder
    root = 'data/tool1'
    resolution = 100       # 'Number of voxels in  z direction.
    mask_volume, ColorTable = generate_mask_from_stl(root, resolution=resolution)
    napari_view_volume(mask_volume)

if __name__ == '__main__':
    main()
