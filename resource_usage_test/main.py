from memory_profiler import memory_usage
import nrrd
from lib.data_preprocess_resource_usage import *
import csv
from csv import writer



def generate_data(ref_path,input_path,slice_count,resize_ratio):

    ref_arr,_=nrrd.read(ref_path)
    ref_arr=ref_arr[:,(ref_arr.shape[1]-slice_count):(ref_arr.shape[1]-1),:]
    input_arr,_=nrrd.read(input_path)
    input_arr=input_arr[:,(input_arr.shape[1]-slice_count):(input_arr.shape[1]-1),:]
    
    if resize_ratio < 1:
        ref_arr=resize_vol(ref_arr, resize_ratio)
        input_arr=resize_vol(input_arr, resize_ratio)
        

    print('Data Size:' +str(ref_arr.shape))
    ref_volume_path='data/tmp/ref_raw.nrrd'
    input_volume_path='data/tmp/input_raw.nrrd'
    nrrd.write(ref_volume_path, ref_arr)
    nrrd.write(input_volume_path, input_arr)
    return ref_volume_path, input_volume_path

def get_volumes_with_memory(ref_path,input_path):
    ref_arr,_=nrrd.read(ref_path)
    input_arr,_=nrrd.read(input_path)
    return ref_arr,input_arr

def registration_operation(ref_arr,input_arr,registration_type,cpu_allocation):

    ref_reg, volume_reg, psnr, rotation_angle, msg_reg, time_req, num_workers= register_3D_volume(rotation_angle=None, max_voxel_count=1000, reference_vol_path=ref_arr,input_vol_path=input_arr,registration_type=registration_type,worker_allocation=cpu_allocation)

    return ref_reg, volume_reg, time_req, psnr, num_workers

def append_list_as_row(file_name, list_of_elem):
    # Open file in append mode
    with open(file_name, 'a+', newline='') as write_obj:
        # Create a writer object from csv module
        csv_writer = writer(write_obj)
        # Add contents of list as last row in the csv file
        csv_writer.writerow(list_of_elem)


def run_performance_test(ref_path, input_path, csv_path, destination_folder, cpu_allocation, slice_increment_count, slice_count_for_resized_operation):

    with open(csv_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Num of Voxels in M", "resize_ratio", "Memory to load 2 volumes", "Time reg_type 1","Memory reg_type 1","PSNR reg_type 1", "Time reg_type 2","Memory reg_type 2","PSNR reg_type 2", "CPU available"])

    ref_arr,_=nrrd.read(ref_path)
    total_slice_count=(ref_arr.shape[1])-1
    resize_ratio_list=[0.2,0.4,0.6,0.8,1]
    slice_increment_count=slice_increment_count
    cpu_allocation=cpu_allocation
    last_max_memory_used=0

    voxel_num=0
    resize_ratio=0
    mem_loading=0
    time_req_type_1=0
    mem_req_type_1=0
    psnr_type_1=0
    time_req_type_2=0
    mem_req_type_2=0
    psnr_type_2=0
    cpu_available=0

    for r in resize_ratio_list:
        print('Doing resized tests')
        device, num_workers, avail_mem=get_machine_processor_memory_Gb()
        print('Available memory:'+ str(avail_mem*1000))
        print('Last Max memory:'+ str(last_max_memory_used))
        resize_ratio=r
        if (avail_mem*1000*0.8)>last_max_memory_used:

            ref_arr_path, input_arr_path= generate_data(ref_path,input_path,slice_count_for_resized_operation, r)
            memory_usage_loading, volumes= memory_usage((get_volumes_with_memory, (ref_arr_path,input_arr_path,), ),retval=True)
            mem_loading=max(memory_usage_loading)

            voxel_num=(volumes[0].shape[0]*volumes[0].shape[1]*volumes[0].shape[2])/1000000.0

            registration_type=1
            memory_usage_registration_1, reg_type_1_res=memory_usage((registration_operation, (volumes[0],volumes[1],registration_type,cpu_allocation,), ),retval=True)
            mem_req_type_1=max(memory_usage_registration_1)
            time_req_type_1=reg_type_1_res[2]
            psnr_type_1=reg_type_1_res[3]

            nrrd.write(destination_folder+'/ref_'+str(int(slice_count_for_resized_operation*r))+'_resized_type_1.nrrd',reg_type_1_res[0])
            nrrd.write(destination_folder+'/input_'+str(int(slice_count_for_resized_operation*r))+'_resized_type_1.nrrd',reg_type_1_res[1])

            del reg_type_1_res

            registration_type=2
            memory_usage_registration_2, reg_type_2_res=memory_usage((registration_operation, (volumes[0],volumes[1],registration_type,cpu_allocation,), ),retval=True)
            mem_req_type_2=max(memory_usage_registration_2)
            time_req_type_2=reg_type_2_res[2]
            psnr_type_2=reg_type_2_res[3]

            cpu_available=reg_type_2_res[4]

            nrrd.write(destination_folder+'/ref_'+str(int(slice_count_for_resized_operation*r))+'_resized_type_2.nrrd',reg_type_2_res[0])
            nrrd.write(destination_folder+'/input_'+str(int(slice_count_for_resized_operation*r))+'_resized_type_2.nrrd',reg_type_2_res[1])

            del reg_type_2_res

            last_max_memory_used=mem_req_type_2

            csv_entry=[voxel_num, resize_ratio, mem_loading, time_req_type_1, mem_req_type_1, psnr_type_1, time_req_type_2, mem_req_type_2, psnr_type_2, cpu_available]
            append_list_as_row(csv_path,csv_entry)

        elif (avail_mem*1000*0.9)>last_max_memory_used:

            ref_arr_path, input_arr_path= generate_data(ref_path,input_path,slice_count_for_resized_operation, r)
            memory_usage_loading, volumes= memory_usage((get_volumes_with_memory, (ref_arr_path,input_arr_path,), ),retval=True)
            mem_loading=max(memory_usage_loading)
            voxel_num=(volumes[0].shape[0]*volumes[0].shape[1]*volumes[0].shape[2])/1000000.0


            registration_type=1
            memory_usage_registration_1, reg_type_1_res=memory_usage((registration_operation, (volumes[0],volumes[1],registration_type,cpu_allocation,), ),retval=True)
            mem_req_type_1=max(memory_usage_registration_1)
            time_req_type_1=reg_type_1_res[2]
            psnr_type_1=reg_type_1_res[3]

            
            nrrd.write(destination_folder+'/ref_'+str(int(slice_count_for_resized_operation*r))+'_resized_type_1.nrrd',reg_type_1_res[0])
            nrrd.write(destination_folder+'/input_'+str(int(slice_count_for_resized_operation*r))+'_resized_type_1.nrrd',reg_type_1_res[1])

            del reg_type_1_res
            # last_max_memory_used=mem_req_type_1
            
            time_req_type_2=0
            mem_req_type_2=0
            psnr_type_2=0
            
            csv_entry=[voxel_num, resize_ratio, mem_loading, time_req_type_1, mem_req_type_1, psnr_type_1, time_req_type_2, mem_req_type_2, psnr_type_2, cpu_available]
            append_list_as_row(csv_path,csv_entry)
        else:
            print('Available memory:'+ str(avail_mem*1000))
            print('Last Max memory:'+ str(last_max_memory_used))


    num_iter=int(total_slice_count/slice_increment_count)
    print(num_iter)
    for iter in range(1,num_iter+1):
        print('Doing in parts tests')
        slices=slice_increment_count*iter
        if slices>total_slice_count:
            slices=total_slice_count
        device, num_workers, avail_mem=get_machine_processor_memory_Gb()
        print('Available memory:'+ str(avail_mem*1000))
        print('Last Max memory:'+ str(last_max_memory_used))

        resize_ratio=1

        if (avail_mem*1000*0.8)>last_max_memory_used:

            ref_arr_path, input_arr_path= generate_data(ref_path,input_path, slices, resize_ratio)
            memory_usage_loading, volumes= memory_usage((get_volumes_with_memory, (ref_arr_path,input_arr_path,), ),retval=True)
            mem_loading=max(memory_usage_loading)
            voxel_num=(volumes[0].shape[0]*volumes[0].shape[1]*volumes[0].shape[2])/1000000.0

            registration_type=1
            memory_usage_registration_1, reg_type_1_res=memory_usage((registration_operation, (volumes[0],volumes[1],registration_type,cpu_allocation,), ),retval=True)
            mem_req_type_1=max(memory_usage_registration_1)
            time_req_type_1=reg_type_1_res[2]
            psnr_type_1=reg_type_1_res[3]

            
            nrrd.write(destination_folder+'/ref_'+str(slices)+'_type_1.nrrd',reg_type_1_res[0])
            nrrd.write(destination_folder+'/input_'+str(slices)+'_type_1.nrrd',reg_type_1_res[1])

            del reg_type_1_res

            registration_type=2
            memory_usage_registration_2, reg_type_2_res=memory_usage((registration_operation, (volumes[0],volumes[1],registration_type,cpu_allocation,), ),retval=True)
            mem_req_type_2=max(memory_usage_registration_2)
            time_req_type_2=reg_type_2_res[2]
            psnr_type_2=reg_type_2_res[3]

            cpu_available=reg_type_2_res[4]

            nrrd.write(destination_folder+'/ref_'+str(slices)+'_type_2.nrrd',reg_type_2_res[0])
            nrrd.write(destination_folder+'/input_'+str(slices)+'_type_2.nrrd',reg_type_2_res[1])

            del reg_type_2_res

            last_max_memory_used=mem_req_type_2
            
            csv_entry=[voxel_num, resize_ratio, mem_loading, time_req_type_1, mem_req_type_1, psnr_type_1, time_req_type_2, mem_req_type_2, psnr_type_2, cpu_available]
            append_list_as_row(csv_path,csv_entry)

        elif (avail_mem*1000*0.9)>last_max_memory_used:

            ref_arr_path, input_arr_path= generate_data(ref_path,input_path, slices, resize_ratio)
            memory_usage_loading, volumes= memory_usage((get_volumes_with_memory, (ref_arr_path,input_arr_path,), ),retval=True)
            mem_loading=max(memory_usage_loading)
            voxel_num=(volumes[0].shape[0]*volumes[0].shape[1]*volumes[0].shape[2])/1000000.0

            registration_type=1
            memory_usage_registration_1, reg_type_1_res=memory_usage((registration_operation, (volumes[0],volumes[1],registration_type,cpu_allocation,), ),retval=True)
            mem_req_type_1=max(memory_usage_registration_1)
            time_req_type_1=reg_type_1_res[2]
            psnr_type_1=reg_type_1_res[3]

            
            nrrd.write(destination_folder+'/ref_'+str(slices)+'_type_1.nrrd',reg_type_1_res[0])
            nrrd.write(destination_folder+'/input_'+str(slices)+'_type_1.nrrd',reg_type_1_res[1])

            del reg_type_1_res


            # last_max_memory_used=mem_req_type_1

            time_req_type_2=0
            mem_req_type_2=0
            psnr_type_2=0

            
            csv_entry=[voxel_num, resize_ratio, mem_loading, time_req_type_1, mem_req_type_1, psnr_type_1, time_req_type_2, mem_req_type_2, psnr_type_2, cpu_available]
            append_list_as_row(csv_path,csv_entry)







if __name__ == '__main__':

    ref_path='data/measurement_1.nrrd'
    input_path='data/measurement_3.nrrd'
    # csv_path= 'registration_resource_usage_test.csv'
    csv_path='/run/user/1002/gvfs/smb-share:server=storage5.local,share=ct_integrity_nvs_only/Jamiul/registration_resource_usage_m1_m3.csv'
    destination_folder='/run/user/1002/gvfs/smb-share:server=storage5.local,share=ct_integrity_nvs_only/Jamiul/processed_volumes_from_resource_test'
    cpu_allocation= 0.60 #percentage of CPU allocated
    slice_increment_count=30
    slice_count_for_resized_operation=200

    run_performance_test(ref_path, input_path, csv_path, destination_folder, cpu_allocation, slice_increment_count, slice_count_for_resized_operation)




