import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt


def get_options():   
    parser = argparse.ArgumentParser(description='Step 2.1 : plot intensity histograms')
    parser.add_argument('--working_dir', action="store", default='/public/workspace/3180111438bit/bmi3_project/Luna2016/',
                        dest="working_dir", type=str)
    parser.add_argument('--out_dir', action="store", default='/public/workspace/3180111438bit/bmi3_project/Luna2016/output_final/',
                        dest="out_dir", type=str)
    parser.add_argument('--figure_dir', action="store", default='/public/workspace/3180111438bit/bmi3_project/Luna2016/2_2_plot_masks',
                        dest="figure_dir", type=str)
    opts = parser.parse_args(sys.argv[1:]) # allow multiple parameter
    return opts

options = get_options()
file_path = options.working_dir
figure_path = options.figure_dir

testMasks = np.load(file_path+'testMasks.npy')
testIm = np.load(file_path+'testImages.npy')
trainIm = np.load(file_path+'trainImages.npy')
trainMasks = np.load(file_path+'trainMasks.npy')
print(testMasks.shape)
print(testIm.shape)
print(trainIm.shape)
print(trainMasks.shape)

for i in range(20):
    fig, (axs1,axs2,axs3) = plt.subplots(1,3, sharey=True)
    axs1.imshow(testIm[i][0], cmap = "gray")
    axs2.imshow(testMasks[i][0], cmap = "gray")
    axs3.imshow(testIm[i][0]*testMasks[i][0], cmap = "gray")
    fig.savefig(figure_path+"/testImage/testImage_"+str(i)+".png")

for i in range(20):
    fig, (axs1,axs2,axs3) = plt.subplots(1,3, sharey = True)
    axs1.imshow(trainIm[i][0], cmap = "gray")
    axs2.imshow(trainMasks[i][0], cmap = "gray")
    axs3.imshow(trainMasks[i][0]*trainIm[i][0], cmap = "gray")
    fig.savefig(figure_path+"/trainImage/trainImage_"+str(i)+".png")

print(np.sum(trainIm[0][0]), np.sum(trainIm[1][0]))
