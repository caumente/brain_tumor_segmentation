# Brain tumor segmentation on magnetic resonance imaging
This project aims to assess automatically each of the regions within a type of brain tumor called Glioma.

To tackle this problem, we have a dataset got from the BraTS challenge organized by MICCAI society. The dataset is 
composed of MRI images of around 370 patients. Four sequences (T1, T1Gd, T2, FLAIR) were taken for each patient, where 
the main difference between them is the type of tissue that is stood out. In addition, the dataset provides a ground 
truth manually labeled by several neuroradiologists, which corresponds to the manual segmentation. Labelling the MR 
images by using the four sequences is a tough task and takes a long time. In this work, it is proposed an artificial 
intelligence architecture to deal with the problem automatically.

Five different architectures have been tested. 3D U-Net, V-Net, and Residual U-Net were used as a baseline since minor 
modifications were done. Then again, Shallow U-Net and Deep U-Net are our proposals. One of the modifications carried 
out, comparing our proposals to the original 3D U-Net, was the replacement of the Batch Normalizations layers by 
Instance Normalizations layers. Moreover, the activation layers ReLU were also replaced by Leaky ReLU. Finally, 
Shallow/Deep U-Net uses four/seven levels of depth and four/seven skip connections instead of the 3 levels and skips 
implemented originally in the 3D U-Net.

