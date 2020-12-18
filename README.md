# Lung Nodule Segmentation Using Deep Learning

Course project for ZJE (Zhejiang University - University of Edinburgh Institute) BMI3 (Biomedical Informatics 3).

We used U-Net (convolutional network) together with Keras (deep learning API) to segment lung nodules from CT images.

Datasets were downloaded from LUNA2016 challenge (https://luna16.grand-challenge.org/Download/).

Group name: 2 Broke Girls.

Group members:

Lanqi WU (lanqi.18@intl.zju.edu.cn)

Jiayi SHEN (jiayis.18@intl.zju.edu.cn)

## Running the code

### Dependencies

Python-3.7.4

pytorch-1.2.0

SimpleITK-2.0.1

numpy-1.19.1

pandas - 1.1.0

argparse - 1.1

Skimage - 0.17.2

Sklearn - 0.23.2

Matplotlib - 3.3.0

keras - 2.2.5

### File Structure

    --bmi3_project
        --LUNA2016
            --1_mask_extraction.py
            --2_segment_lung_ROI.py
            --2_1_plots_histogram.py
            --2_2_plot_masks.py
            --3_unet.py
            --3_1_testing.py
            --subset[0-4]
            --annotations.csv
            --output_final
                --trainImages.npy
                --trainMasks.npy
                --testImages.npy
                --testMasks.npy
                --[All processed images and masks (.npy)]
                --line_plot.jpg
            --figures
                --trainImage
            --prediction
                --[All prediction image](.jpg)
            --unet.log
            --unet.hdf5

### Pre-Processing

- Binary Thresholding
- Selecting the three largest connected regions
- Erosion to separate nodules attached to blood vessels
- Dilation to keep nodules attached to the lung walls
- Filling holes by dilation

```
1_mask_extraction.py 
    [--subset]   (default: -1)
    [--working_dir]  (default: /bmi3_project/Luna2016/)
    [--out_dir]  (default: /bmi3_project/Luna2016/output_final/)
```
```
2_segment_lung_ROI.py
    [--working_dir]  (default: /bmi3_project/Luna2016/)
```

### U-NET

```
3_unet.py 
    [--out_dir]  (default: /bmi3_project/Luna2016/output_final/)
    [--prediction_fig]  (default: /bmi3_project/Luna2016/prediction/)
    [--epochs]  (default: 500)
    [--batch_size]  (default: 8)
    [--lr]  (default: 0.001)
    [--filter_width]  (default: 3)
    [--start_filters]  (default: 32)
```

```
3_1_testing.py
    [--working_dir]  (default: /bmi3_project/Luna2016/)
    [--out_dir]  (default: /bmi3_project/Luna2016/output_final/)
    [--lr]  (default: 0.001)
    [--filter_width]  (default: 3)
    [--start_filters]  (default: 32)
```