import numpy as np
import nrrd
import os
from scipy import ndimage

from dipy.segment.mask import median_otsu, bounding_box, crop

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


def rotateImage(img, angle, pivot):
    padX = [img.shape[1] - pivot[0], pivot[0]]
    padY = [img.shape[0] - pivot[1], pivot[1]]
    imgP = np.pad(img, [padY, padX], 'constant')
    imgR = ndimage.rotate(imgP, angle, reshape=False, order=1)
    return imgR[padY[0] : -padY[1], padX[0] : -padX[1]]

def get_bin_mask(vol_arr):
    b0_mask, mask_vol = median_otsu(vol_arr, median_radius=0, numpass=1)
    zero_arr=np.zeros(mask_vol.shape)
    mask_vol=mask_vol+zero_arr
    return mask_vol


def bg_remove_complete(vol_arr):
    
    b0_mask, mask_vol = median_otsu(vol_arr, median_radius=0, numpass=1)

    
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
            for j in range(vol_arr.shape[2]):
                if mask[i][j]==0:
                    new_img[i][j]=0
        new_img_list.append(new_img)
    
    volume_array=np.stack(new_img_list, axis=1)
    
    return volume_array

def bg_remove_outer(vol_arr):
    
    b0_mask, mask_vol = median_otsu(vol_arr, median_radius=0, numpass=1)
    
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
                   
    
def resize_vol(vol_arr, resize_ratio):
    result = ndimage.zoom(vol_arr, resize_ratio, order=1)
    return result


def cal_iou(static,moving):
    
#     component1 = [0 if i ==0 else 1 for i in static]
#     component2 = [0 if i ==0 else 1 for i in moving]
#     component1 = static
#     component2 = moving

    component1 = np.array(static, dtype=bool)
    component2 = np.array(moving, dtype=bool)

    overlap = component1*component2 # Logical AND
    union = component1 + component2 # Logical OR

    IOU = overlap.sum()/float(union.sum()) # Treats "True" as 1,
                                           # sums number of Trues
                                           # in overlap and union
                                           # and divides
                
    return IOU


def comp_iou(static,moving):
    overlap_count=0
    union_count=0
    for i in range(static.shape[0]):
        for j in range(static.shape[1]):
            if static[i][j]!=0:
                if moving[i][j]!=0:
                    overlap_count=overlap_count+1
                    
    for i in range(static.shape[0]):
        for j in range(static.shape[1]):
            if static[i][j]!=0:
                if moving[i][j]==0:
                    union_count=union_count+1
                    
    for i in range(static.shape[0]):
        for j in range(static.shape[1]):
            if static[i][j]==0:
                if moving[i][j]!=0:
                    union_count=union_count+1
                    
    union_count=union_count+overlap_count
                    
    iou=0.0
    iou=overlap_count/union_count
    
    return iou


def register_2DCT(template_data,moving_data,angle_step):
    
#     regtools.overlay_images(template_data, moving_data, 'Static', 'Overlay', 'ori')
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
    IOU_val=0
    reg_transform= None
    new_img=moving_data
    
    for angle in range(-180,180,angle_step):
        
        
        transform=TranslationTransform2D()
        TranslationTransform=affreg.optimize(template_data, moving_data, transform, params0)

        

        TranslationTransformed = TranslationTransform.transform(moving_data,'linear')
        
        TranslationTransformed=rotateImage(TranslationTransformed,angle,[int(moving_data.shape[0]/2),int(moving_data.shape[1]/2)])
        
                                           
        transform = TranslationTransform2D()
        TranslationTransform = affreg.optimize(template_data, TranslationTransformed, transform, params0)


                                           

#         TranslationTransformed = TranslationTransform.transform(TranslationTransformed)

        transform=RigidTransform2D()
        rigid = affreg.optimize(template_data, TranslationTransformed, transform, params0,starting_affine=TranslationTransform.affine)

        
        transform=AffineTransform2D()
        AffineTransform = affreg.optimize(template_data, TranslationTransformed, transform, params0,starting_affine=rigid.affine)

        AffineTransformed = AffineTransform.transform(TranslationTransformed,'linear')
        
        new_IOU=cal_iou(template_data,AffineTransformed.astype(np.uint16))
#         new_IOU=comp_iou(template_data,template_data)
        
        print(new_IOU)
        print(angle)
        
        if new_IOU>IOU_val:
            rot_angle=angle
            IOU_val=new_IOU
#             reg_transform=AffineTransform
            new_img=AffineTransformed
            
        
    # regtools.overlay_images(template_data, new_img, 'Static', 'Overlay', 'transformed')
    print(rot_angle)

    return rot_angle

    
def register_vol(ref_arr,vol_arr,angle_step):
    import random
    
    rot_angle=0
    
    template_data = ref_arr
    template_data = np.transpose(ref_arr, (1, 0, 2))
    template_affine = np.eye(4)

    moving_data = vol_arr
    moving_data = np.transpose(vol_arr, (1, 0, 2))
    moving_affine = np.eye(4)
    
#     print(template_data.shape)

    
#     template_data_bin=get_bin_mask(template_data)
#     moving_data_bin=get_bin_mask(moving_data)
    
#     b0_mask, mask_vol = median_otsu(template_data, median_radius=0, numpass=1)
#     template_data_bin=b0_mask
    
#     b0_mask, mask_vol = median_otsu(moving_data, median_radius=0, numpass=1)
#     moving_data_bin=b0_mask






    template_data_bin=(template_data)
    moving_data_bin=(moving_data)
    
    rot_angle_list=[]
    ref_arr_list=[]
    vol_arr_list=[]
    
    ref_arr_list=[]
    vol_arr_list=[]
    for i in range(0,int(template_data_bin.shape[0])):
        ref_arr_list.append(template_data_bin[i,:,:])
        vol_arr_list.append(moving_data_bin[i,:,:]) 
    
    split_size=5
    vol_size = int(template_data_bin.shape[0]/split_size)
    for i in range(split_size):
        idx = random.randint(i*vol_size, (i+1)* vol_size)
#         idx = random.randint(i*vol_size, (i+1)* vol_size)
        # rotate the selected slice
        rot_angle_list.append(register_2DCT(ref_arr_list[idx],vol_arr_list[idx],angle_step))



    import statistics
    median_angle = statistics.median(rot_angle_list)
    print(rot_angle_list)


    
    rot_angle=median_angle
    print(rot_angle)
    





    # The mismatch metric
    nbins = 32
    sampling_prop = None
    metric = MutualInformationMetric(nbins, sampling_prop)

    # The optimization strategy
    level_iters = [100, 100, 100]
    sigmas = [3.0, 1.0, 0.0]
    factors = [4, 2, 1]

    affreg = AffineRegistration(metric=metric,
                                level_iters=level_iters,
                                sigmas=sigmas,
                                factors=factors)
    params0 = None
    
#     regtools.overlay_slices(template_data, moving_data, None, 0,
#                         "Static", "Moving")
    
    # align all 2d slices
    transform = TranslationTransform3D()
    translation = affreg.optimize(template_data, moving_data, transform, params0,
                                  template_affine, moving_affine)
    transformed = translation.transform(moving_data)
    
    # regtools.overlay_slices(template_data, transformed, None, 0,
    #                     "Static", "before_rot")
    
    #rotation of volume
    new_data = ndimage.rotate(moving_data, rot_angle, axes=(1,2), reshape=False, order=1)
    
    moving_data=new_data.astype(np.uint16)
    
    # regtools.overlay_slices(template_data, moving_data, None, 0,
    #                     "Static", "after_rot")

    # tranlate to avoid shift after rotation
    transform = TranslationTransform3D()
    translation = affreg.optimize(template_data, moving_data, transform, params0,
                                  template_affine, moving_affine)
    
    transformed = translation.transform(moving_data)
    # regtools.overlay_slices(template_data, transformed, None, 0,
    #                     "Static", "after_2")
    
#     TranslationTransform3D,
#     RigidTransform3D,
#     AffineTransform3D,
#     RigidScalingTransform3D,
#     RotationTransform3D,
#     ScalingTransform3D,
#     RigidIsoScalingTransform3D,
    
    transform = RigidTransform3D()
    rigid = affreg.optimize(template_data, moving_data, transform, params0,
                            template_affine, moving_affine,
                            starting_affine=translation.affine)
    
#     transform = ScalingTransform3D()
#     scaling = affreg.optimize(template_data, moving_data, transform, params0,
#                             template_affine, moving_affine,
#                             starting_affine=rigid.affine)
    
    
#     transform = RotationTransform3D()
#     rotation = affreg.optimize(template_data, moving_data, transform, params0,
#                             template_affine, moving_affine,
#                             starting_affine=scaling.affine)
    
    
    affreg.level_iters = [10000, 1000, 100]

    transform = AffineTransform3D()
    affine = affreg.optimize(template_data, moving_data, transform, params0,
                            template_affine, moving_affine,
                            starting_affine=rigid.affine)
    
    transformedaffine = affine.transform(moving_data)
    
    # regtools.overlay_slices(template_data, transformedaffine, None, 0,
    #                     "Static", "after_affine")
    
    new_vol_arr=np.transpose(transformedaffine, (1, 0, 2))
    
    # nrrd.write('m5wrt1.nrrd',new_vol_arr)

    
    return new_vol_arr, rot_angle

def pre_proc_3d(ref_path,vol_path,bg_remove_mode=2,resize_ratio=1,crop_point=0, pre_proc_ref=False,angle_step=5):
    # ref_arr, ref_header = nrrd.read(ref_path)    
    # vol_arr, vol_header = nrrd.read(vol_path)
    
    ref_arr=ref_path    
    vol_arr=vol_path

    # return ref_arr, vol_arr


    if pre_proc_ref:
        if crop_point>0:
            ref_arr=ref_arr[:,:crop_point,:]
            
        if resize_ratio!=1:
            ref_arr=resize_vol(ref_arr, resize_ratio)
        
        if bg_remove_mode==0:
            ref_arr=get_bin_mask(ref_arr)

        if bg_remove_mode==1:
            ref_arr=bg_remove_complete(ref_arr)

        if bg_remove_mode==2:
            ref_arr=bg_remove_outer(ref_arr)

        if bg_remove_mode==3:
            ref_arr=ref_arr        
        
        
        
    
    if crop_point>0:
        vol_arr=vol_arr[:,:crop_point,:]
        
    if resize_ratio!=1:
        vol_arr=resize_vol(vol_arr, resize_ratio) 
        
    if bg_remove_mode==0:
        vol_arr=get_bin_mask(vol_arr)
        
    if bg_remove_mode==1:
        vol_arr=bg_remove_complete(vol_arr)
        
    if bg_remove_mode==2:
        vol_arr=bg_remove_outer(vol_arr)
        
    if bg_remove_mode==3:
        vol_arr=vol_arr
    
#     mins, maxs = bounding_box(vol_arr)
#     vol_chopped = crop(vol_arr, mins, maxs)
        
    registered_vol, angle_of_rot=register_vol(ref_arr,vol_arr,angle_step)
    
    print(cal_iou(ref_arr,registered_vol))

    return ref_arr, registered_vol
    
        
    



