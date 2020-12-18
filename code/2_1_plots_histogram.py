import argparse
import sys
import numpy as np
from glob import glob
import matplotlib.pyplot as plt

def get_options():   
    parser = argparse.ArgumentParser(description='Step 2.1 : plot intensity histograms')
    parser.add_argument('--out_dir', action="store", default='/public/workspace/3180111438bit/bmi3_project/Luna2016/output_final/',
                        dest="out_dir", type=str)
    parser.add_argument('--figure_dir', action="store", default='/public/workspace/3180111438bit/bmi3_project/Luna2016/2_1_plots/',
                        dest="figure_dir", type=str)
    opts = parser.parse_args(sys.argv[1:]) # allow multiple parameter
    return opts

options = get_options()
file_list=glob(options.out_dir+"images_*.npy")
count = 0
for img_file in file_list:
    count = count + 1
    imgs_to_process = np.load(img_file).astype(np.float64) 
    print("on image", img_file)
    for i in range(len(imgs_to_process)):
        img = imgs_to_process[i]
        mean = np.mean(img)
        std = np.std(img)
        img = img-mean
        img = img/std
        fig, (axs1,axs2) = plt.subplots(2)
        axs1.imshow(img, cmap = "gray")
        axs2.hist(img.flatten(),bins=200, color = "gray", range=[-3,3])
        fig.savefig(options.figure_dir+"ROI_step1_hist"+str(count)+"_"+str(i)+".png")