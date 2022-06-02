
# importing the required libraries
import os
import random
import shutil
import sys
from tkinter import N
from weakref import ref
import cv2
import time
from glob import glob
import nibabel as nib
import numpy as np
from PIL import Image
import json
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import QDateTime, Qt, QTimer,pyqtSignal,QEvent
from PyQt5 import   uic
from pathlib import Path
import nrrd
from numpy.core.fromnumeric import size
from numpy.core.numeric import full
from scipy import ndimage
from dipy.segment.mask import median_otsu, bounding_box, crop
import math
from multiprocessing import Pool
from skimage.metrics import structural_similarity as ssim

 
from dipy.viz import regtools
from dipy.data import fetch_stanford_hardi
from dipy.data.fetcher import fetch_syn_data
from dipy.io.image import load_nifti
from dipy.align.imaffine import (transform_centers_of_mass,
                                 AffineMap,
                                 MutualInformationMetric,
                                 AffineRegistration)
from dipy.align.transforms import (TranslationTransform3D,
                                   RigidTransform3D,
                                   AffineTransform3D,
                                  RigidScalingTransform3D,
                                  RotationTransform3D,
                                  ScalingTransform3D,
                                  RigidIsoScalingTransform3D)
 
from dipy.align.transforms import (TranslationTransform2D,
                                   RigidTransform2D,
                                   AffineTransform2D,
                                  RotationTransform2D,
                                  RigidIsoScalingTransform2D,
                                  ScalingTransform2D,
                                  RigidScalingTransform2D)
from math import log10, sqrt
import matplotlib.pyplot as plt





def rotateImage(img, angle, pivot):
    padX = [img.shape[1] - pivot[0], pivot[0]]
    padY = [img.shape[0] - pivot[1], pivot[1]]
    imgP = np.pad(img, [padY, padX], 'constant')
    imgR = ndimage.rotate(imgP, angle, reshape=False, order=1)
    return imgR[padY[0] : -padY[1], padX[0] : -padX[1]]

def clean_background_outer(vol_arr):
    b0_mask, mask_vol = median_otsu(vol_arr, median_radius=0, numpass=1)
   
    # nrrd.write('b0_mask.nrrd', b0_mask)
 
    zero_arr=np.zeros(mask_vol.shape)
    mask_vol=mask_vol+zero_arr
   
    mask_list=[]
    for i in range(0,vol_arr.shape[1]):
        mask_list.append(mask_vol[:,i,:])
       
    vol_img_list=[]
    for i in range(0,vol_arr.shape[1]):
        vol_img_list.append(vol_arr[:,i,:])
   
    new_img_list=[]
    for w in range(0,vol_arr.shape[1]):
        new_img=vol_img_list[w]
        mask=mask_list[w]
 
        for i in range(vol_arr.shape[0]):
            arr=np.zeros(mask.shape[1])
            for j in range(mask.shape[1]):
                arr[j]=mask[i][j]
            _index = np.where(arr != 0)
            start=99990
            end=0
            for t in _index:
                for l in t:
                    if(start>l):
                        start=l
                    end=l
            for j in range(vol_arr.shape[2]):
                if j< start:
                    new_img[i][j]=0
                else:
                    if j>end:
                        new_img[i][j]=0
 
        new_img_list.append(new_img)
                   
    final_new_img_list=[]
    for w in range(0,vol_arr.shape[1]):
        new_img=new_img_list[w]
        mask=mask_list[w]
 
        new_img=rotateImage(new_img,90,[int(new_img.shape[0]/2),int(new_img.shape[1]/2)])
        mask=rotateImage(mask,90,[int(mask.shape[0]/2),int(mask.shape[1]/2)])
 
        for i in range(vol_arr.shape[0]):
            arr=np.zeros(mask.shape[1])
            for j in range(mask.shape[1]):
                arr[j]=mask[i][j]
            _index = np.where(arr != 0)
            start=99990
            end=0
            for t in _index:
                for l in t:
                    if(start>l):
                        start=l
                    end=l
            for j in range(vol_arr.shape[2]):
                if j< start:
                    new_img[i][j]=0
                else:
                    if j>end:
                        new_img[i][j]=0
 
        new_img=rotateImage(new_img,-90,[int(new_img.shape[0]/2),int(new_img.shape[1]/2)])
 
        final_new_img_list.append(new_img)
                   
    volume_array=np.stack(final_new_img_list, axis=1)
   
    return volume_array

def clean_background_outer_Parallel(vol_arr):
    #multiprocessing for faster operation 6 worker
    pool = Pool()
 
    chunk_num=6
    if vol_arr.shape[1]<chunk_num+1:
        volume_bs=clean_background_outer(vol_arr)
        return vol_arr
    else:
        chunk_length=int(math.floor((vol_arr.shape[1]/chunk_num)))

    result1 = pool.apply_async(clean_background_outer, [vol_arr[:,:chunk_length,:]])  
    result2 = pool.apply_async(clean_background_outer, [vol_arr[:,((chunk_length*1)):chunk_length*2,:]])
    result3 = pool.apply_async(clean_background_outer, [vol_arr[:,((chunk_length*2)):chunk_length*3,:]])
    result4 = pool.apply_async(clean_background_outer, [vol_arr[:,((chunk_length*3)):chunk_length*4,:]])
    result5 = pool.apply_async(clean_background_outer, [vol_arr[:,((chunk_length*4)):chunk_length*5,:]])
    result6 = pool.apply_async(clean_background_outer, [vol_arr[:,((chunk_length*5)):vol_arr.shape[1],:]])
    
    answer1 = result1.get()
    answer2 = result2.get()
    answer3 = result3.get()
    answer4 = result4.get()
    answer5 = result5.get()
    answer6 = result6.get()
     
    volome_bs=np.concatenate((answer1,answer2,answer3,answer4,answer5,answer6),axis=1)
    # napari_viewer_reg_confirmation(volome_bs,volome_bs)
    return volome_bs

def resize_vol(vol_arr, resize_ratio):
    result = ndimage.zoom(vol_arr, resize_ratio, order=1)
    return result

def cal_ssim(img, img_noise):
    ssim_noise = ssim(img, img_noise)
    return abs(ssim_noise)

def register_2DCT(template_data,moving_data,angle_step):
   
    # The mismatch metric
    nbins = 32
    sampling_prop = None
    metric = MutualInformationMetric(nbins, sampling_prop)
   
    # The optimization strategy
    level_iters = [1000, 100, 100]
    sigmas = [3.0, 1.0, 0.0]
    factors = [4, 2, 1]
 
    affreg = AffineRegistration(metric=metric,
                                level_iters=level_iters,
                                sigmas=sigmas,
                                factors=factors)
 
    params0 = None
   
    rot_angle=0
    ssim_val=0
   
    for angle in range(-180,180,angle_step):
       
        transform=TranslationTransform2D()
        TranslationTransform=affreg.optimize(template_data, moving_data, transform, params0)
        TranslationTransformed = TranslationTransform.transform(moving_data,'linear')

        TranslationTransformed=rotateImage(TranslationTransformed,angle,[int(moving_data.shape[0]/2),int(moving_data.shape[1]/2)])
                                      
        transform = TranslationTransform2D()
        TranslationTransform = affreg.optimize(template_data, TranslationTransformed, transform, params0)
#         TranslationTransformed = TranslationTransform.transform(TranslationTransformed)
        # transform=RigidTransform2D()
        # rigid = affreg.optimize(template_data, TranslationTransformed, transform, params0,starting_affine=TranslationTransform.affine)
 
        # transform=AffineTransform2D()
        # AffineTransform = affreg.optimize(template_data, TranslationTransformed, transform, params0,starting_affine=rigid.affine)
        # AffineTransformed = AffineTransform.transform(TranslationTransformed,'linear')

        new_test=TranslationTransform.transform(TranslationTransformed,'linear')
       
        # new_ssim=cal_ssim(template_data,AffineTransformed.astype(np.uint16))
        new_ssim=cal_ssim(template_data,new_test.astype(np.uint16))
       
        print(new_ssim)
        print(angle)
       
        if new_ssim>ssim_val:
            rot_angle=angle
            ssim_val=new_ssim
            # new_img=AffineTransformed

    print(rot_angle)
    return rot_angle

def get_estimated_angle(template_data,moving_data,angle_step_size,min_size):
    rot_angle=0

    resize_ratio=min_size/np.max([template_data.shape[0],template_data.shape[1],template_data.shape[2]])

    template_data_resized=resize_vol(template_data,resize_ratio)
    moving_data_resized=resize_vol(moving_data,resize_ratio)

    template_data_resized=clean_background_outer_Parallel(template_data_resized)
    moving_data_resized=clean_background_outer_Parallel(moving_data_resized)


    template_data_resized = np.transpose(template_data_resized, (1, 0, 2))
 
    moving_data_resized = np.transpose(moving_data_resized, (1, 0, 2))


    

    rot_angle_list=[]
    ref_arr_list=[]
    vol_arr_list=[]

    ref_arr_list=[]
    vol_arr_list=[]
    for i in range(0,int(template_data_resized.shape[0])):
        ref_arr_list.append(template_data_resized[i,:,:])
        vol_arr_list.append(moving_data_resized[i,:,:])

    if template_data_resized.shape[0]<7:
        rot_angle=register_2DCT(ref_arr_list[int(template_data_resized.shape[0]/2)],vol_arr_list[int(template_data_resized.shape[0]/2)],angle_step_size)
        return rot_angle

    split_size=7
    vol_size = int(template_data_resized.shape[0]/split_size)

    pool=Pool()

    i=0
    idx1 = random.randint(i*vol_size, (i+1)* vol_size)

    i=1
    idx2 = random.randint(i*vol_size, (i+1)* vol_size)

    i=2
    idx3 = random.randint(i*vol_size, (i+1)* vol_size)

    i=3
    idx4 = random.randint(i*vol_size, (i+1)* vol_size)

    i=5
    idx5 = random.randint(i*vol_size, (i+1)* vol_size)

    i=6
    idx6 = random.randint(i*vol_size, (i+1)* vol_size)

    result1 = pool.apply_async(register_2DCT,[ref_arr_list[idx1],vol_arr_list[idx1],angle_step_size])  
    result2 = pool.apply_async(register_2DCT,[ref_arr_list[idx2],vol_arr_list[idx2],angle_step_size])   
    result3 = pool.apply_async(register_2DCT,[ref_arr_list[idx3],vol_arr_list[idx3],angle_step_size])
    result4 = pool.apply_async(register_2DCT,[ref_arr_list[idx4],vol_arr_list[idx4],angle_step_size])
    result5 = pool.apply_async(register_2DCT,[ref_arr_list[idx5],vol_arr_list[idx5],angle_step_size])
    result6 = pool.apply_async(register_2DCT,[ref_arr_list[idx6],vol_arr_list[idx6],angle_step_size])


    answer1 = result1.get()
    answer2 = result2.get()
    answer3 = result3.get()
    answer4 = result4.get()
    answer5 = result5.get()
    answer6 = result6.get()

    rot_angle_list.append(answer1)
    rot_angle_list.append(answer2)
    rot_angle_list.append(answer3)
    rot_angle_list.append(answer4)
    rot_angle_list.append(answer5)
    rot_angle_list.append(answer6)

    import statistics
    median_angle = statistics.median(rot_angle_list)
    print(rot_angle_list)

    rot_angle=median_angle
    print('Estimated rotation angle:' +str(rot_angle))

    return rot_angle

## NAPARI 3D Viewer
def napari_viewer_reg_confirmation(img3D_ref, img3D_faulty):
    import napari
    napari.gui_qt()
    # viewer = napari.view_image(img3D_pos, colormap='magma')
    # %gui qt
    viewer = napari.Viewer(ndisplay=3)
 
    viewer.add_image(img3D_ref,name='reference scan', colormap='gray')#colormap='gist_earth')
    viewer.add_image(img3D_faulty,name='Input scan', colormap='red',opacity=0.6)#colormap='gist_earth')
    napari.run(force=True)
    return viewer

def queryBox(query):
    msg = QMessageBox()
    msg.setIcon(QMessageBox.Information)
    msg.setText(query)
    msg.setWindowTitle("Confirmation!!")
    msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
    retval = msg.exec_()
    if retval == 65536:
        print(" Clicked NO ")
        return '0'
    else: 
        print(" Clicked Yes ")
        return '1'

def showdialog_Registration_Validation():
    msg = QMessageBox()
    msg.setIcon(QMessageBox.Information)
    msg.setText("Are you satisfied by this registration performance?")
    msg.setInformativeText("If you select yes, the result will be saved for future learning. \n\nIf you select No, the registration will be repeated. \n\n Click cancel to stop registration")
    msg.setWindowTitle("Visual  validation")
    # msg.setDetailedText("If you select No, the registration will be repeated.")
    msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel)
    retval = msg.exec_()
    # print(retval)
    if retval == 65536:
        print(" Clicked NO ")
        return '0'
    elif retval == 16384: 
        print(" Clicked Yes ")
        return '1'
    else:
        print('Registration cancelled')
        return '2'

def cal_psnr(original,toComp):
    mse = np.mean((original - toComp) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = np.max([np.max(original),np.max(toComp)])
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr

def register_volume(template_data,moving_data,registration_type,chunk_size,rot_angle,resize_ratio=1):

    if resize_ratio!=1:
        template_data=resize_vol(template_data,resize_ratio)
        moving_data=resize_vol(moving_data,resize_ratio)

    template_data = clean_background_outer_Parallel(template_data)
    moving_data = clean_background_outer_Parallel(moving_data)

    template_data = np.transpose(template_data, (1, 0, 2))
    template_affine = np.eye(4)
    moving_data = np.transpose(moving_data, (1, 0, 2))
    moving_affine = np.eye(4)

    if registration_type==1:
        chunk_size=chunk_size
        if chunk_size>moving_data.shape[0]:
            chunk_size=moving_data.shape[0]
    elif registration_type==2:
        voxel_num=(moving_data.shape[0]*moving_data.shape[1]*moving_data.shape[2])/1000000
        if voxel_num>100:
            factor=voxel_num/30
            ratio=1/(factor**(1/3))
            template_data=resize_vol(template_data,ratio)
            moving_data=resize_vol(moving_data,ratio)
            chunk_size=moving_data.shape[0]

            # print('Chunk Size: '+chunk_size)
            print(moving_data.shape)

        else:
            chunk_size=moving_data.shape[0]
    else:
        print('Please specify preprocess type.')
        return 0,0,0
    
    print('registering')
    # The mismatch metric
    nbins = 32
    sampling_prop = None
    metric = MutualInformationMetric(nbins, sampling_prop)
 
    # The optimization strategy
    level_iters = [1000, 100, 100]
    sigmas = [3.0, 1.0, 0.0]
    factors = [4, 2, 1]
 
    affreg = AffineRegistration(metric=metric,
                                level_iters=level_iters,
                                sigmas=sigmas,
                                factors=factors)
    params0 = None
   
    moving_data_full=moving_data
    template_data_full=template_data

    part_list=[]    
    partition_size=int(math.floor(moving_data_full.shape[0]/chunk_size))

    part1=moving_data_full[:chunk_size,:,:]
    part1=np.pad(part1, ((5,5), (0,0), (0, 0)), 'constant')

    for i in range(1,partition_size):
        tmp_part=moving_data_full[i*chunk_size:chunk_size*(i+1),:,:]
        tmp_part=np.pad(tmp_part, ((5,5), (0,0), (0, 0)), 'constant')
        part_list.append(tmp_part)
    
    tmp_part=moving_data_full[chunk_size*partition_size:,:,:]
    tmp_part=np.pad(tmp_part, ((5,5), (0,0), (0, 0)), 'constant')
    part_list.append(tmp_part)

    moving_data=part1
    template_data=template_data_full[:chunk_size,:,:]
    template_data=np.pad(template_data, ((5,5), (0,0), (0, 0)), 'constant')

    # align all 2d slices
    transform = TranslationTransform3D()
    translation = affreg.optimize(template_data, moving_data, transform, params0,
                                  template_affine, moving_affine)
    initial_translation=translation
    moving_data = translation.transform(moving_data)

   
    #rotation of volume
    moving_data = (ndimage.rotate(moving_data, rot_angle, axes=(1,2), reshape=False, order=1)).astype(np.uint16)
   
 
    # tranlate to avoid shift after rotation
    transform = TranslationTransform3D()
    translation = affreg.optimize(template_data, moving_data, transform, params0,
                                  template_affine, moving_affine)
   
    transform = RigidTransform3D()
    rigid = affreg.optimize(template_data, moving_data, transform, params0,
                            template_affine, moving_affine,
                            starting_affine=translation.affine)
   
    affreg.level_iters = [10000, 1000, 100]
 
    transform = AffineTransform3D()
    affine = affreg.optimize(template_data, moving_data, transform, params0,
                            template_affine, moving_affine,
                            starting_affine=rigid.affine)

    affreg.level_iters = [1000, 100, 100]
    transform = TranslationTransform3D()
    translation2 = affreg.optimize(template_data, moving_data, transform, params0,
                            template_affine, moving_affine,
                            starting_affine=affine.affine)
   
    # moving_data = affine.transform(moving_data)
    moving_data = translation2.transform(moving_data)

    full_tool=moving_data[5:chunk_size+5]

    counter=2

    for arr in part_list:

        print('doing part: '+str(counter))
        tmp=initial_translation.transform(arr)
        tmp=(ndimage.rotate(tmp, rot_angle, axes=(1,2), reshape=False, order=1)).astype(np.uint16)
        tmp = affine.transform(tmp)
        tp2=tmp[5:arr.shape[0]-5]
        full_tool=np.concatenate((full_tool,tp2),axis=0)
        counter=counter+1

        del tmp
        del tp2

    
    full_tool=np.transpose(full_tool, (1, 0, 2))
    template_data_full=np.transpose(template_data_full, (1, 0, 2))
    psnr=cal_psnr(template_data_full,full_tool)

    return template_data_full, full_tool, psnr

def normalize_vol(ref_arr, input_arr):
    input_arr=(input_arr-np.min(input_arr)) / (np.max(input_arr)-np.min(input_arr))
    input_arr=input_arr*np.max(ref_arr)
    return ref_arr, input_arr

def simpleRek2Py(filename, image_width, image_height, image_depth, voxel_datatype):
    
    '''
    filename: path to the rek file
    
    image_width x image_height x image_depth: the dimension to be resized
    500x500x500 - 0.2 GB file
    1000x1000x1000 - 1.4 GB file
    2000x2000x2000 - 15.5 GB file

    voxel_datatype: the datatype of the file
    uint16 - integer data file
    float32 - float data file
    '''
    print('\n Opening rek file: %s\n  - size=(%d,%d,%d) \n  - voxel_datatype=%s '%(filename, image_width, image_height, image_depth,voxel_datatype))
    
    if (voxel_datatype == "uint16"):
        datatype = np.uint16
    elif (voxel_datatype == "float32"):
        datatype = np.float32
    else:
        raise ValueError("Unsupported datatype")

    with open(filename, 'rb') as fd:
        raw_file_data = fd.read()        
    image = np.frombuffer(raw_file_data[2048:], dtype=datatype)
    shape = image_width, image_height, image_depth

    return image.reshape(shape)

def VGI2Py(file_path):
    import os
    # Using readlines()
    vgi_file =file_path + '.vgi'
    if not os.path.exists(vgi_file):
        print(' Error: The VGI file is not found: \n', vgi_file)

        return 0
    else:
        file1 = open(vgi_file, 'r')
        Lines = file1.readlines()
        count = 0
        # Strips the newline character
        for line in Lines:
            count += 1
            line_str= line.strip()
            terms = line_str.split(' ')
            # print("Line{}: {} \n {}".format(count, line_str, terms ))

            if terms[0]=='size':
                size=(int(terms[2]), int(terms[3]), int(terms[4]))
                print(' size = ', size)
            elif terms[0]=='bitsperelement':
                if terms[2]=='8':
                    voxel_type = np.uint8

                elif terms[2]=='16':
                    voxel_type = np.uint16

                else:
                    print(' Voxel type is not an usual value = ', terms[2])

            elif terms[0]=='SkipHeader':
                SkipHeader=int(terms[2])
                print(' SkipHeader = ', SkipHeader)
            
        # load the BCM volume
        voxel_count = size[0] * size[1] * size[2]
        f = open(file_path,'rb') #only opens the file for reading
        vol_arr=np.fromfile(f,dtype=voxel_type,offset=SkipHeader,count=voxel_count)
        f.close()
        vol_arr=vol_arr.reshape(size[0],size[1],size[2])
        return vol_arr

def load_volume( path):
        import numpy as np
        # Draw rek rek filesL
        filename, file_extension = os.path.splitext(path)
        if os.path.exists(path):
            # read 3D volume from rek file
            if file_extension=='.rek':
                # volume_array = scanner.load_Rek2Py(rek_file)
                # enter slice dimensions (width x height x depth)
                image_width = 500
                image_height = 500
                image_depth = 500
                # enter voxel datatype ("float32" or "uint16")
                voxel_datatype = "uint16"
                # read 3D volume from rek file
                
                volume_array = simpleRek2Py(path, image_width, image_height, image_depth, voxel_datatype)  
            
            elif file_extension=='.bd': 

                volume_array = VGI2Py(path)
  

            elif file_extension=='.nii':
                import nibabel as nib
                volume_array = nib.load(path).get_data()


            elif file_extension=='.nrrd':
                import nrrd
                volume_array, _ = nrrd.read(path)

                
            else: 
                msg = '\n Warning: The file format : %s is not supported!!!!'%(path)
                raise ValueError(msg)

        else:
            msg = '\n Warning: The file : %s is not found!!!!'%(path)
            raise Exception(msg)
        # # sanity check the volume mush have this shape [N, M, N]
        # if len(np.unique(volume_array.shape))<=2:
        #     if volume_array.shape[0]!=volume_array.shape[2]:
        #         volume_array = volume_array.transpose(0, 2, 1)
        # else: 
        #     print('Error: the volume does not respect the shape critereon [N, M, N]!')
        #     return -1

        return volume_array

def load_3D_from_2D(path):
    slices = []
    img_arr=[]
    img_count=1
 
    for f in sorted(os.listdir(path)):
        img = Image.open(str(path)+'/'+f)
        slices.append(img)
        img_arr.append(np.array(img))
        print(f'[+] Slice read: {img_count}', end='\r')
        img_count=1+img_count
    print()
    #stack the 2D arrays over one another to generate the 3D array
    volume_array=np.stack(img_arr, axis=1)
    return volume_array

def get_scan_folders(root_folder, pattern, ext = '.tiff'):
    from glob import glob 
    import os
    import numpy as np
    list_paths= glob(os.path.join(root_folder,  '**'), recursive=True)
    scans_list = []
    for path_ in list_paths:
        if pattern in path_ :
            if ext in path_:
                scans_list.append(os.path.dirname(path_))
    scans_list = list( np.unique( np.array(scans_list) ) )
    return scans_list

def load_raw_data(path):
    if os.path.exists(path):
        if os.path.isdir(path):
            slices_path=str(get_scan_folders(path,'Rec',ext='.tiff')[0])
            ref_volume_arr=load_3D_from_2D(slices_path)
            return ref_volume_arr
            
        elif os.path.isfile(path):
            data = load_volume(path)
            return data
    else:
        print(path)
        msg = f'\n\nError: the path [{path}] cannot be found!!!!\nplease recheck the configuration file config/config.yml\n\n'
        raise Exception(msg)

def load_input_data(file):
    try: 
        file.shape
        # Assign the array 
        arr = np.copy(file)
    except:
        # load path file
        arr = load_raw_data(file)
    return arr




def register_3D_volume(reference_vol_path, input_vol_path, registration_type=1, chunk_size = 30, rotation_angle=None, angle_step_size=1, min_size = 100, normalize = False):

    try:
        print('loading data...')
        ref_arr=load_input_data(reference_vol_path)
        input_arr=load_input_data(input_vol_path)

        #check dimension if equal or not


        print('starting preprocessing...')
        # ref_reg, volume_reg, psnr=register_volume(ref_arr,input_arr,registration_type,chunk_size,11)

        # napari_viewer_reg_confirmation(ref_arr,input_arr)
        if normalize:
            ref_arr, input_arr=normalize_vol(ref_arr,input_arr)
        
        if rotation_angle==None:
            val='n'
            while(val!='y'):
                # tic = time.perf_counter()
                
                rot_angle=get_estimated_angle(ref_arr,input_arr,angle_step_size,min_size)
                resize_ratio=min_size/np.max([ref_arr.shape[0],ref_arr.shape[1],ref_arr.shape[2]])

                reg_type=2
                # register the volume with reference
                ref_reg, volume_reg, psnr=register_volume(ref_arr,input_arr,reg_type,chunk_size,rot_angle,resize_ratio=resize_ratio)            
                
                # toc = time.perf_counter() - tic
                # print(toc)
                #napari view
                viewer = napari_viewer_reg_confirmation(ref_reg, volume_reg)
                from_dialog= showdialog_Registration_Validation()
                if from_dialog=='0':
                    val='n'
                elif from_dialog=='1':
                    val='y'
                    rotation_angle=rot_angle
                else:
                    return np.zeros([3,3,3]),np.zeros([3,3,3]),0,0,'Registration cancelled.'
                if registration_type==3:
                    return np.zeros([3,3,3]),np.zeros([3,3,3]),0,rotation_angle,'angle derived.'

            print('Redistration parameter accepted!! Registering main volume')


        ref_reg, volume_reg, psnr=register_volume(ref_arr,input_arr,registration_type,chunk_size,rotation_angle)
        msg_reg='Registration completed.'
    except Exception as e: # work on python 3.x
        print('Failed operation: '+ str(e))
        return np.zeros([3,3,3]),np.zeros([3,3,3]),0,0,'Registration failed.'


    return ref_reg, volume_reg, psnr, rotation_angle, msg_reg