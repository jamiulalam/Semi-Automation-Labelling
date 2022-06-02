
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
    '''
    Rotate Image data according to <angle> and <pivot> point.
    '''
    padX = [img.shape[1] - pivot[0], pivot[0]]
    padY = [img.shape[0] - pivot[1], pivot[1]]
    imgP = np.pad(img, [padY, padX], 'constant')
    imgR = ndimage.rotate(imgP, angle, reshape=False, order=1)
    return imgR[padY[0] : -padY[1], padX[0] : -padX[1]]

def clean_background_outer(vol_arr):
    '''
    Cleans outer backgound of volume data.
    '''
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

def clean_background_outer_Parallel(vol_arr,  worker_allocation=0.80):

    '''

    Cleans outer backgound of volume data dividing into several segments and cleaning them parallelly

    '''
    device, num_workers, avail_mem=get_machine_processor_memory_Gb()
    worker_to_use=int(num_workers*worker_allocation)
    chunk_num=worker_to_use
    if vol_arr.shape[1]<chunk_num+1:
        volume_bs=clean_background_outer(vol_arr)
        return vol_arr
    else:
        chunk_length=int(math.floor((vol_arr.shape[1]/chunk_num)))



    pool = Pool()

    result_list=[]
    last_idx=0
    for i in range(chunk_num):
        last_idx=chunk_length*(i+1)
        if i==chunk_num-1:
            last_idx=vol_arr.shape[1]
        result_list.append(pool.apply_async(clean_background_outer, [vol_arr[:,((chunk_length*i)):last_idx,:]]))

    answer_list=[]
    for result in result_list:
        answer_list.append(result.get())


    volome_bs=np.concatenate(answer_list,axis=1)
    # napari_viewer_reg_confirmation(volome_bs,volome_bs)

    return volome_bs

def resize_two_volumes(ref_volume, input_volume):
    resize_ratio_all = [x/y for x, y in zip(ref_volume.shape,input_volume.shape)]
    resize_ratio_list = np.unique(resize_ratio_all)
    if len(resize_ratio_list)==1:
        resize_ratio = resize_ratio_list[0]
        ref_volume_rz = resize_vol(ref_volume, resize_ratio)
        input_volume_rz = resize_vol(input_volume, resize_ratio)
        return ref_volume_rz, input_volume_rz
    else:
        print(f' the input volumes dont have the same axis scale: {resize_ratio_all}')
        return -1, -1

def resize_vol(vol_arr, resize_ratio):
    result = ndimage.zoom(vol_arr, resize_ratio, order=1)
    return result

def cal_ssim(img, img_noise):
    '''
    Calculate SSIM between 2 images.
    '''
    ssim_noise = ssim(img, img_noise)
    return abs(ssim_noise)

def register_2DCT(template_data,moving_data,angle_step, reg_optimize_factors):

    '''Get rotational difference angle between 2D input image <moving_data> according to reference <template_data>.
    '''
   

    # The mismatch metric
    nbins = reg_optimize_factors["nbins"]
    sampling_prop = reg_optimize_factors["sampling_prop"]
    metric = MutualInformationMetric(nbins, sampling_prop)
   
    # The optimization strategy
    level_iters = reg_optimize_factors["level_iters"]
    sigmas = reg_optimize_factors["sigmas"]
    factors = reg_optimize_factors["factors"]

    
 
    affreg = AffineRegistration(metric=metric,
                                level_iters=level_iters,
                                sigmas=sigmas,
                                factors=factors,
                                verbosity=0)
 
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

        new_test=TranslationTransform.transform(TranslationTransformed,'linear')
       
        new_ssim=cal_ssim(template_data,new_test.astype(np.uint16))

        if new_ssim>ssim_val:
            rot_angle=angle
            ssim_val=new_ssim

    print(f'\n - SSIM={new_ssim} , rot_angle={rot_angle} ')
    return rot_angle

def get_estimated_angle(template_data,moving_data,angle_step_size,min_size,num_of_split=7, worker_allocation=0.80, \
    reg_optimize_factors={}):

    '''
    Estimate the rotational difference of input <moving_data> wrt a reference volume <template_data>
    \ntemplate_data = reference volume array 
    \nmoving_data = input volume array 
    \nangle_step_size=1
    \nmin_size = 100
    \nworker_allocation=0.80
    \nreg_optimize_factors=reg_optimize_factors

    '''

    rot_angle=0

    resize_ratio=min_size/np.max([template_data.shape[0],template_data.shape[1],template_data.shape[2]])

    template_data_resized=resize_vol(template_data,resize_ratio)
    moving_data_resized=resize_vol(moving_data,resize_ratio)


    template_data_resized=clean_background_outer_Parallel(template_data_resized, worker_allocation=worker_allocation)
    moving_data_resized=clean_background_outer_Parallel(moving_data_resized, worker_allocation=worker_allocation)


    template_data_resized = np.transpose(template_data_resized, (1, 0, 2))
 
    moving_data_resized = np.transpose(moving_data_resized, (1, 0, 2))

    ref_arr_list=[]
    vol_arr_list=[]
    for i in range(0,int(template_data_resized.shape[0])):
        ref_arr_list.append(template_data_resized[i,:,:])
        vol_arr_list.append(moving_data_resized[i,:,:])


    device, num_workers, avail_mem=get_machine_processor_memory_Gb()
    worker_to_use=int(num_workers*worker_allocation)

    split_size=num_of_split

    if template_data_resized.shape[0]<split_size:
        #too little volume to divide into different segments, so get angle from 1 slice
        rot_angle=register_2DCT(ref_arr_list[int(template_data_resized.shape[0]/2)],vol_arr_list[int(template_data_resized.shape[0]/2)],angle_step_size, reg_optimize_factors)
        return rot_angle

    vol_size = math.floor(template_data_resized.shape[0]/split_size)
    index_list=[]
    for i in range(split_size):
        last_num=(i+1)* vol_size
        if last_num> template_data_resized.shape[0]:
            last_num=template_data_resized.shape[0]
        index_list.append(random.randint((i*vol_size)+1, last_num-1))

    print('Slices selected for angle estimation: '+ str(index_list))

    answer_list=[]

    num_of_step=math.ceil(split_size/worker_to_use)

    for i in range(num_of_step):
        pool=Pool()
        result_list=[]
        if (worker_to_use*(i+1))>(len(index_list)):
            last_idx=len(index_list)
        else:
            last_idx=worker_to_use*(i+1)

        for idx in index_list[i*worker_to_use:last_idx]:
            result_list.append(pool.apply_async(register_2DCT,[ref_arr_list[idx],vol_arr_list[idx],angle_step_size,reg_optimize_factors]))
        for result in result_list:
            answer_list.append(result.get())

    import statistics
    median_angle = statistics.median(answer_list)
    print('Estimated rotation angle list: '+ str(answer_list))

    rot_angle=median_angle
    print('Estimated rotation angle:' +str(rot_angle))

    return rot_angle

## NAPARI 3D Viewer
def napari_viewer_reg_confirmation(img3D_ref, img3D_faulty):
    '''
    Displays 2 volumes overlapping each other
    '''

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
        print(" Not Accepted ")
        return '0'
    else: 
        print(" Accepted ")
        return '1'

def showdialog_Registration_Validation():
    msg = QMessageBox()
    msg.setIcon(QMessageBox.Information)
    msg.setText("Are you satisfied by this registration performance?")
    msg.setInformativeText("If you select yes, the result will be saved for future use. \n\nIf you select No, the registration will be repeated. \n\n Click cancel to stop registration")
    msg.setWindowTitle("Visual  validation")
    # msg.setDetailedText("If you select No, the registration will be repeated.")
    msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel)
    retval = msg.exec_()
    # print(retval)
    if retval == 65536:
        print(" Not Accepted ")
        return '0'
    elif retval == 16384: 
        print(" Accepted ")
        return '1'
    else:
        print('Registration cancelled')
        return '2'

def cal_psnr(original,toComp):
    '''
    Calculate PSNR between to volumes or images'''
    mse = np.mean((original - toComp) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = np.max([np.max(original),np.max(toComp)])
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr

def register_volume(template_data,moving_data,registration_type,chunk_size,rot_angle,resize_ratio=1, max_voxel_count=30,\
    worker_allocation=0.80, reg_optimize_factors={}):
    '''
    Register input volume data according to the reference volume using configurations from parameters.
    '''


    if resize_ratio!=1:
        print(f'\n - Volumes resizing .. ')
        template_data=resize_vol(template_data,resize_ratio)
        moving_data=resize_vol(moving_data,resize_ratio)

    print(f'\n - Cleanning the outer background of the CT scan... ')
    template_data = clean_background_outer_Parallel(template_data, worker_allocation=worker_allocation)
    moving_data = clean_background_outer_Parallel(moving_data, worker_allocation=worker_allocation)

    template_data = np.transpose(template_data, (1, 0, 2))
    template_affine = np.eye(4)
    moving_data = np.transpose(moving_data, (1, 0, 2))
    moving_affine = np.eye(4)

    if registration_type==1:
        if chunk_size>moving_data.shape[0]:
            chunk_size=moving_data.shape[0]
    elif registration_type==2:
        voxel_num=(moving_data.shape[0]*moving_data.shape[1]*moving_data.shape[2])/1000000
        if voxel_num>max_voxel_count:
            factor=voxel_num/max_voxel_count
            ratio=1/(factor**(1/3))
            print(f'\n - Volumes resizing with ratio= {ratio}.. ')
            template_data=resize_vol(template_data,ratio)
            moving_data=resize_vol(moving_data,ratio)
            chunk_size=moving_data.shape[0]

            print('Chunk Size: ', chunk_size)
            print(moving_data.shape)

        else:
            chunk_size=moving_data.shape[0]
    else:
        print('Please specify preprocess type.')
        return 0,0,0
    
    print(f'\n Starting registeration using  {chunk_size} chunks ...')
    # The mismatch metric
    nbins = reg_optimize_factors["nbins"]
    sampling_prop = reg_optimize_factors["sampling_prop"]
    metric = MutualInformationMetric(nbins, sampling_prop)
 
    # The optimization strategy
    level_iters= reg_optimize_factors["level_iters"]
    sigmas = reg_optimize_factors["sigmas"]
    factors = reg_optimize_factors["factors"]
    print(f'\n - AffineRegistration parameter: \n \t - level_iters={level_iters}\n \t - sigmas={sigmas} \
                 \n \t - factors={factors}')
    affreg = AffineRegistration(metric=metric,
                                level_iters=level_iters,
                                sigmas=sigmas,
                                factors=factors,
                                verbosity=0)
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
    print(f'\n Aligning all 2D slices...')
    transform = TranslationTransform3D()
    translation = affreg.optimize(template_data, moving_data, transform, params0,
                                  template_affine, moving_affine)
    initial_translation=translation
    moving_data = translation.transform(moving_data)

   
    #rotation of volume
    print(f'\n Rotating the volumes...')
    moving_data = (ndimage.rotate(moving_data, rot_angle, axes=(1,2), reshape=False, order=1)).astype(np.uint16)

    # tranlate to avoid shift after rotation
    print(f'\n Tranlating the volume to avoid shift after rotation...')
    transform = TranslationTransform3D()
    translation = affreg.optimize(template_data, moving_data, transform, params0,
                                  template_affine, moving_affine)
   
    print(f'\n Rigid Transformation of volume...')
    transform = RigidTransform3D()
    rigid = affreg.optimize(template_data, moving_data, transform, params0,
                            template_affine, moving_affine,
                            starting_affine=translation.affine)
   
    affreg.level_iters = [i * 10 for i in level_iters]
 
    print(f'\n Affine Transformation of volume...')
    transform = AffineTransform3D()
    affine = affreg.optimize(template_data, moving_data, transform, params0,
                            template_affine, moving_affine,
                            starting_affine=rigid.affine)

    affreg.level_iters = level_iters
    transform = TranslationTransform3D()
    translation2 = affreg.optimize(template_data, moving_data, transform, params0,
                            template_affine, moving_affine,
                            starting_affine=affine.affine)
   
    moving_data = translation2.transform(moving_data)

    full_tool=moving_data[5:chunk_size+5]

    counter=2
    print(f'\n - Concatenating the finalized registred chunks')
    for arr in part_list:
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
        # sanity check the volume mush have this shape [N, M, N]
        if len(np.unique(volume_array.shape))<=2:
            if volume_array.shape[0]!=volume_array.shape[2]:
                volume_array = volume_array.transpose(0, 2, 1)
        else: 
            print('Error: the volume does not respect the shape critereon [N, M, N]!')
            return -1

        return volume_array

def load_3D_from_2D(path):
    '''
    Create 3D volume from tiif image slices.
    '''
    slices = []
    img_arr=[]
    img_count=1
 
    for f in sorted(os.listdir(path)):
        img = Image.open(str(path)+'/'+f)
        slices.append(img)
        img_arr.append(np.array(img))
        print(f'[+] Slice read: {img_count}', end='\r')
        img_count=1+img_count
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
            
            # slices_path=str(get_scan_folders(path,'Rec',ext='.tiff')[0])
            slices_path=path
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
    '''

    load data from path ot file

    '''
    try: 
        file.shape
        # Assign the array 
        arr = np.copy(file)
    except:
        print('loading from folder')
        # load path file
        arr = load_raw_data(file)
    return arr

def get_machine_processor_memory_Gb(disp=0):
    '''
    Get machine resoource information.'''
    # get CPU/GPU torch device, workers, and the available Memory in Gb
    import psutil
    mem = psutil.virtual_memory()
    avail_mem = mem.available/1000000000
    import torch
    if  torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    # number pf processes
    import multiprocessing
    num_workers = multiprocessing.cpu_count()
    # display
    if disp>=1:
        print(f'\n - Available Processor =  {device} / {num_workers} workers \n - Available Memory =  {avail_mem} Gb \n - All memory info :   {mem} ')
    return device, num_workers, avail_mem

def register_3D_volume(reference_vol_path, input_vol_path, registration_type=1, chunk_size = 30, \
                      rotation_angle=None, angle_step_size=1, min_size = 100, normalize = False, max_voxel_count=30, worker_allocation=0.60,\
                           num_of_split=7, reg_optimize_factors = {'nbins':32, 'sampling_prop':None, 'level_iters':[1000,100,100], 'sigmas':[3.0, 1.0, 0.0], 'factors':[4, 2, 1]}):
    '''
    register a input volume <input_vol_path> wrt a reference volume <reference_vol_path>
    \n
    \n registration_type : 1- fast registration 2- slow but more accurate  3- angle estimation
    \n chunk_size = 30 #num of slices to register at once for registration type 1
    \n rotation_angle=None #rotational difference between reference and input volume
    \n angle_step_size=1 # step size to calculate rotational difference
    \n min_size = 100 #size of volume to resize for initial validation registration
    \n normalize = False #normalize the input volumes
    \n max_voxel_count = 30 #maximum voxel number in M to register at once for regitration type 2
    \n worker_allocation = 0.80 #percentage of cpu worker to use
    \n num_of_split = 7 #num of slices to consider for angle estimation
    \n reg_optimize_factors= {'nbins':32, 'sampling_prop':None, 'level_iters':[1000,100,100], 
    \n                          'sigmas':[3.0, 1.0, 0.0], 'factors':[4, 2, 1]}


    '''

    device, num_workers, avail_mem=get_machine_processor_memory_Gb()
    worker_to_use=int(num_workers*worker_allocation)
    print('Number of worker allocated: '+str(worker_to_use))
    
    print('\n\n ==> Starting  registration and preprocessing:')
    # Timing start
    tic = time.perf_counter()
    try:
        print('\n - Loading data...')
        ref_arr=load_input_data(reference_vol_path)
        input_arr=load_input_data(input_vol_path)
        print('Data loaded: '+str(ref_arr.shape))

        # napari_viewer_reg_confirmation(ref_arr,input_arr)
        if normalize:
            ref_arr, input_arr=normalize_vol(ref_arr,input_arr)
        
        if rotation_angle==None:
            print('\n - Estimating rotational difference...')
            val='n'
            while(val!='y'):
                
                rot_angle=get_estimated_angle(ref_arr,input_arr,angle_step_size,min_size, num_of_split=num_of_split, worker_allocation=worker_allocation, reg_optimize_factors=reg_optimize_factors)
                                
                resize_ratio=min_size/np.max([ref_arr.shape[0],ref_arr.shape[1],ref_arr.shape[2]])
                reg_type=2
                print(f'\n --> step1 : registing the resized version :\
                     \n \t - resize_ratio={resize_ratio} \
                     \n \t - reg_type={reg_type}')
                # register the volume with reference
                ref_reg, volume_reg, psnr=register_volume(ref_arr,input_arr,reg_type,chunk_size,rot_angle,resize_ratio=resize_ratio,max_voxel_count=max_voxel_count, worker_allocation=worker_allocation, reg_optimize_factors=reg_optimize_factors)            
                
                # toc = time.perf_counter() - tic
                # print(toc)
                #napari view
                viewer = napari_viewer_reg_confirmation(ref_reg, volume_reg)
                # from_dialog='1'
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

            print('Registration parameter accepted!! Registering main volume')

        print(f'\n --> step2 : registing the full sized volume :\
                     \n \t - rotation_angle={rotation_angle} \
                     \n \t - chunk_size={chunk_size} \
                     \n \t - reg_type={registration_type}')

        ref_reg, volume_reg, psnr=register_volume(ref_arr,input_arr,registration_type,chunk_size,rotation_angle,max_voxel_count=max_voxel_count, worker_allocation=worker_allocation, reg_optimize_factors=reg_optimize_factors)
        msg_reg='Registration completed.'
    except:
        print(f'\nWarning: Registration failed...')
        return np.zeros([3,3,3]),np.zeros([3,3,3]),0,0,'Registration failed.'

    # toc = time.perf_counter() - tic
    # toc = int(toc*100)/100
    # msg_reg=  msg_reg+  f'\n- rotation_angle={rotation_angle}' + \
    #           f'\n- chunk_size={chunk_size}' +\
    #           f'\n- reg_type={reg_type}'+ \
    #           f'\n- psnr={psnr}'+ \
    #           f'\n- Execution time = ' + str(toc) + ' seconds ' + str(int(toc//60)) + ' Minutes\n' 
    print(msg_reg)
    return ref_reg, volume_reg, psnr, rotation_angle, msg_reg

    

######################## Example of function execution  ############################
def run_3D_preprocessing_inspections():
    ref_path= 'data/m1.nrrd'
    input_path= 'data/m2_53.nrrd'

    ref_reg, volume_reg, psnr, rotation_angle, msg_reg=register_3D_volume(reference_vol_path=input_path, input_vol_path=ref_path, registration_type=1, worker_allocation=0.50)
    nrrd.write('ref.nrrd',ref_reg)
    nrrd.write('input.nrrd',volume_reg)
if __name__ == '__main__':
    run_3D_preprocessing_inspections()