import os
import numpy as np
import matplotlib.pyplot as plt

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def cropping_tool_lengh2(volume, disp=1):

    '''
    cropeing the volume along the axis=1 to remove \
    the suporting object used duting the CT scanning 
    '''

    from numpy import linalg as LA
    from scipy.signal import argrelextrema
    img_proj_y = np.max(volume, axis=2)
    row_std_list = []
    x_list = range(img_proj_y.shape[1] -1 )
    for x in x_list:
        row_std = np.std(img_proj_y[:,x]) 
        # err= img_proj_y[:,x] - img_proj_y[:,x+1]
        # row_std = LA.norm(err)
        row_std_list.append(row_std)
        # print( 'srd %d = %s' %(x,row_std) )
    import matplotlib.pyplot as plt
    row_std = smooth(np.array(row_std_list) ,20)
    row_std_diff = np.diff(row_std_list)
    # local maxima

    raw_maxima = argrelextrema(row_std, np.greater)
    # local minima
    raw_minima = argrelextrema(row_std, np.less)
    # crop the image 
    idx0 = raw_minima[0][0]-3
    idx1 = raw_maxima[0][-1] +3
    volume=volume[ :, idx0:idx1, :] 

    # display
    if disp>=1:
        #print the outputs
        # print(f'\n local minima =  {raw_minima}\n local maxima =  {raw_maxima} ')
        print(f'\v volume size = {volume.shape}')
        print(f'\n  idx0={idx0}, idx1={idx1} ')
        print(f'\v volume croped size = {volume.shape}')
        # plot
        plt.plot(row_std_list, 'r')
        plt.plot(row_std_diff, 'g')
        plt.ylabel('slices std')
        plt.ylabel('slice')
        plt.show()
        # show image
        show_image(img_proj_y , 'projection',  figsize = (8,8))
    return volume

def show_image(img, img_title, cmap="cividis", figsize = (8,8)):
    # show image
    fig = plt.figure(figsize = figsize) # create a 5 x 5 figure 
    ax3 = fig.add_subplot(111)
    ax3.imshow(img, interpolation='none', cmap=cmap)
    ax3.set_title(img_title)#, fontsize=40)
    # plt.savefig('./residual_image.jpg')   
    plt.axis("off")
    plt.show()

def homogenity_slices(volume, type='ssim'):
    vec_homog = []
    for i in range(volume.shape[1]):
        img= volume[:,i,:]
        if i > 0:
            img=volume[:,i,:]
            # compute homoginity
            if type =='ssim':# SSIM
                from skimage.metrics import structural_similarity as ssim
                vec_homog.append(ssim(img,img_old))
            else: # MSE
                vec_homog.append(np.mean(np.abs(img-img_old)))
        img_old = img
    return vec_homog

def crop_volumes(volume_ref, volume_input):
	#  croping the volume slices (remove slices of undesired part: support object, etc)
	if volume_ref.shape[1]>400 and volume_input.shape[1]>400 :
			print(f'\n - Cropping the volumes is in progress...') 
			volume_ref   = cropping_tool_lengh2(volume_ref, disp=0)
			volume_input = cropping_tool_lengh2(volume_input, disp=0)
	else:
			print(f'\n - Cropping the volumes is ignored because of the samll size of the volume= {volume_input.shape[1]} slices!')
	# make sure the volume have the same size
	L_min = np.min([volume_ref.shape[1],volume_input.shape[1]])
	volume_ref, volume_input = volume_ref[:,-L_min:,:], volume_input[:,-L_min:,:]
	return volume_ref, volume_input

if __name__ == '__main__':
	volume_ref=[]
	volume_input=[]
	#  croping the volume slices (remove slices of undesired part: support object, etc)
	volume_ref, volume_input = crop_volumes(volume_ref, volume_input)