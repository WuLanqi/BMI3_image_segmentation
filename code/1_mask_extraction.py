from __future__ import print_function, division
import SimpleITK as sitk
import numpy as np
from glob import glob
import pandas as pd
import os
import argparse
import sys


# generate wait bar
try:
    from tqdm import tqdm
except:
    print('Error...')
    tqdm = lambda x: x
    
def get_options():   
    parser = argparse.ArgumentParser(description='Step 1: mask extraction')
    parser.add_argument('--subset', action="store", default=-1, dest = "subset", type = int)
    parser.add_argument('--working_dir', action="store", default='/public/workspace/3180111438bit/bmi3_project/Luna2016/',
                        dest="working_dir", type=str)
    parser.add_argument('--out_dir', action="store", default='/public/workspace/3180111438bit/bmi3_project/Luna2016/output_final/',
                        dest="out_dir", type=str)
    opts = parser.parse_args(sys.argv[1:]) # allow multiple parameter
    return opts

def make_mask(center,diam,z,width,height,spacing,origin):
    '''
    center : centers of circles (px) -- list of coordinates (x,y,z)
    diam : diameters of circles (px)
    z : z position of slice in world coordinates (mm)
    width & height : pixel dimension of the image (px)
    spacing : (mm/px) conversion rate -- np array (x,y,z)
    origin : (mm) -- np.array (x,y,z)
    '''
    ################################## BELOW ########################################
    ##### code referece: https://www.kaggle.com/c/data-science-bowl-2017/overview/tutorial #########
    # Fill in 0 everywhere in the image
    mask = np.zeros([height,width])
    # Define the voxel range where the nodule is
    v_center = (center-origin)/spacing
    v_diam = int(diam/spacing[0]+5)
    v_xmin = np.max([0,int(v_center[0]-v_diam)-5])
    v_xmax = np.min([width-1,int(v_center[0]+v_diam)+5])
    v_ymin = np.max([0,int(v_center[1]-v_diam)-5])
    v_ymax = np.min([height-1,int(v_center[1]+v_diam)+5])
    v_xrange = range(v_xmin,v_xmax+1)
    v_yrange = range(v_ymin,v_ymax+1)
    # Fill in 1 within nodule sphere
    for v_x in v_xrange:
        for v_y in v_yrange:
            p_x = spacing[0]*v_x + origin[0]
            p_y = spacing[1]*v_y + origin[1]
            if np.linalg.norm(center-np.array([p_x,p_y,z]))<=diam:
                mask[int((p_y-origin[1])/spacing[1]),int((p_x-origin[0])/spacing[0])] = 1.0
    ################################## ABOVE ########################################
    return(mask)

def matrix2int16(matrix):
    ''' 
    matrix must be a numpy array NXN
    Returns uint16 version
    '''
    m_min= np.min(matrix)
    m_max= np.max(matrix)
    matrix = matrix-m_min
    return(np.array(np.rint( (matrix-m_min)/float(m_max-m_min) * 65535.0),dtype=np.uint16))

def get_filename(file_list, case):
    for f in file_list:
        if case in f:
            return(f)
            
if __name__ == '__main__':
    # Getting list of image files
    options = get_options()
    luna_path = options.working_dir
    luna_subset_path = luna_path+"subset"+str(options.subset)+"/"
    output_path = options.out_dir
    file_list=glob(luna_subset_path+"*.mhd")
    # Load the locations of the nodes (annotation)
    df_node = pd.read_csv(luna_path+"annotations.csv")
    df_node["file"] = df_node["seriesuid"].map(lambda file_name: get_filename(file_list, file_name))
    df_node = df_node.dropna()
    ################################## BELOW ########################################
    ##### code referece: https://www.kaggle.com/c/data-science-bowl-2017/overview/tutorial #########
    for fcount, img_file in enumerate(tqdm(file_list)):
        mini_df = df_node[df_node["file"]==img_file] #get all nodules associate with file
        if mini_df.shape[0]>0: # skip those without nodules
            # load the data
            itk_img = sitk.ReadImage(img_file) 
            img_array = sitk.GetArrayFromImage(itk_img) # indexes are (z,y,x)
            num_z, height, width = img_array.shape # notice the order
            origin = np.array(itk_img.GetOrigin()) # origin : (x,y,z) (mm)
            spacing = np.array(itk_img.GetSpacing()) # spacing
            # go through all nodes
            for node_idx, cur_row in mini_df.iterrows():       
                node_x = cur_row["coordX"]
                node_y = cur_row["coordY"]
                node_z = cur_row["coordZ"]
                diam = cur_row["diameter_mm"]
                # keep 3 slices
                imgs = np.ndarray([3,height,width],dtype=np.float32)
                masks = np.ndarray([3,height,width],dtype=np.uint8)
                center = np.array([node_x, node_y, node_z])   # nodule center
                v_center = np.rint((center-origin)/spacing)  # nodule center in voxel space (x,y,z)
                for i, i_z in enumerate(np.arange(int(v_center[2])-1, int(v_center[2])+2).clip(0, num_z-1)): # clip prevents going out of bounds in Z
                    mask = make_mask(center, diam, i_z*spacing[2]+origin[2], width, height, spacing, origin)
                    masks[i] = mask
                    imgs[i] = img_array[i_z]
                np.save(os.path.join(output_path,"images_%04d_%04d.npy" % (fcount, node_idx)),imgs)
                np.save(os.path.join(output_path,"masks_%04d_%04d.npy" % (fcount, node_idx)),masks)
    ################################## ABOVE ########################################