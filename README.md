# BTS U-Net: A Data-Driven Approach to Brain Tumor Segmentation


## Overview
BTS U-Net is a novel lightweight deep learning architecture for brain tumor segmentation in magnetic resonance 
imaging (MRI). The model prioritizes efficiency and sustainability without sacrificing performance, making 
it ideal for scenarios with limited computational resources. This repository includes the code and results 
from the study, which outperformed some traditional architectures in key metrics while maintaining computational 
simplicity. BTS U-Net reduces training time by up to 79.2% while achieving comparable 
performance, with DICE scores of 0.811, 0.878, and 0.908 for the region Enhancing Tumor (ET), 
Tumor Core (TC), and Whole Tumor (WT), respectively. Additionally, the study highlights the importance of 
glioma grading for segmentation accuracy, suggesting that a two-step approach—classification followed by segmentation—yields better results.


The main contributions of the work are the following:

- We propose a novel efficient lightweight architecture that outperforms some of the most popular architectures previously used in biomedical image segmentation, such as U-Net [7] or V-Net [8]. Our model offers significantly lower training and computational requirements while achieving comparable performance.

- The study shows statistical differences between LGG and HGG, and suggests a potential shift in glioma segmentation strategies to optimize outcomes, especially for HGG tumors, the most aggressive ones.

- We prove that the brain tumor segmentation problem can be effectively approached in a 2-step way by  first classifying the type of glioma between HGG and LGG, and then segmenting the MRI


## Methodology
### Data
The model was trained on the BraTS 2020 dataset, which contains multi-modal MRI scans (T1, T1-Gd, T2, and FLAIR)
of gliomas. The dataset was split into training (80%), validation (10%), and test (10%) subsets, with 
appropriate preprocessing including voxel normalization, cropping, and augmentation.

<p align="center">
   <img src="https://github.com/caumente/brain_tumor_segmentation/blob/main/paper_imgs/MRI_sequences.png" width="400">
</p>

### Model Architecture
BTS U-Net builds on the U-Net architecture with the following enhancements:
1. **Extra Skip Connection:** Captures multi-level features.
2. **Instance Normalization:** Improves memory efficiency and handles variability in input distributions.
3. **Leaky ReLU Activation:** Addresses the dying ReLU problem.
4. **Deep Supervision:** Incorporates intermediate layers into the loss calculation for better performance.

These features make BTS U-Net both efficient and effective, using significantly fewer parameters than comparable models.

![BTS U-Net](./paper_imgs/BTS%20U-Net.jpg)


## Results


### MRI characteristics and masks analysis

The analysis of MRI sequences revealed that the pixel intensity distributions are significantly different between
High-Grade Gliomas (HGG) and Low-Grade Gliomas (LGG). These findings align with medical knowledge, as HGG 
typically have more distinct features due to their aggressive and infiltrative nature. LGG, being slower-growing
and less infiltrative, exhibit subtler changes in MRI characteristics, making them harder to distinguish.

![BTS U-Net](./paper_imgs/histograms.png)

The analysis of segmentation masks provided by radiologists further reinforced the findings from the MRI characteristics study.

<p align="center">
   <img src="https://github.com/caumente/brain_tumor_segmentation/blob/main/paper_imgs/boxplot_voxels_labels_v2.png" width="400">
</p>

### Comparative Performance

BTS U-Net was compared against established architectures like U-Net, V-Net, and nnU-Net. Key findings include:
- **Model size:** Reduced the number of trainable parameters by at least half.
- **Training Time:** Up to 79.2% faster than other architectures.
- **Performance:** Comparable performance with significantly fewer parameters (5M vs. nnU-Net's 33M).


| Model         | No. Parameters | Size (MB) | FLOPs (10¹²) | Training Time/Epoch (s) | Val Cohort - ET | Val Cohort - TC | Val Cohort - WT |
|---------------|:--------------:|:---------:|:------------:|:-----------------------:|:---------------:|:---------------:|:---------------:|
| 3D U-Net      |      16M       |   72.8    |     5.17     |          301            |      0.698      |      0.783      |      0.877      |
| V-Net         |      45M       |   174.0   |     2.13     |          187            |      0.635      |      0.699      |      0.867      |
| SegResNet     |      11M       |   40.3    |     0.91     |          228            |      0.776      |      0.832      |      0.893      |
| nnU-Net       |      33M       |   127.3   |    50.80     |          230            |      0.791      |      0.843      |      0.903      |
| **BTS U-Net** |       5M       |   18.8    |     1.41     |          168            |      0.790      |      0.841      |      0.901      |


Qualitative results are displayed below:

![Qualitative results](./paper_imgs/qualitative_colors.png)


## Conclusions

This study highlights the effectiveness of BTS U-Net, a lightweight deep learning model for brain tumor segmentation, 
offering competitive accuracy with reduced computational demands. Its efficient design makes it ideal for 
resource-constrained applications, such as medical devices.

Significant differences between low-grade (LGG) and high-grade gliomas (HGG) were identified, emphasizing the need for
tumor-specific strategies. Training separate models for HGG improved segmentation performance, supporting a two-step 
approach of classification followed by segmentation.

BTS U-Net demonstrates a practical and sustainable solution for AI-driven medical imaging, with potential for broader 
clinical applications and future research on dataset variability and generalization.

## References

- B. H. Menze, A. Jakab, S. Bauer, J. Kalpathy-Cramer, K. Farahani,  J. Kirby, Y. Burren, N. Porz, J. Slotboom, R. Wiest, et al., The multimodal brain tumor image segmentation benchmark (BRATS), IEEE Transactions on Medical Imaging 34 (10) (2014) 1993–2024
- O. Ronneberger, P. Fischer, T. Brox, U-Net: Convolutional networks for biomedical image segmentation, in: International Conference on Medical Image Computing and Computer-Assisted Intervention, Springer, 2015, pp. 234–24
- F. Isensee, P. F. Jaeger, S. A. Kohl, J. Petersen, K. H. Maier-Hein, nnU-net: a self-configuring method for deep learning-based biomedical image segmentation, Nature methods 18 (2) (2021) 203–211
