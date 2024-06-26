# BTS U-Net: A data-driven approach to brain tumor segmentation through deep learning
This work aims to propose a novel approach to deal with brain tumor segmentation through the use of artificial 
intelligence techniques.

To tackle this problem, we used a dataset got from the BraTS challenge organized by MICCAI society among others. The 
dataset is composed of around 370 magnetic resonance images taken from patients who suffered from a specific type of
brain tumor called glioma. A total of four sequences (T1, T1Gd, T2, FLAIR) were provided in each MRI, and the main 
difference between them is the type of tissue that is stood out. In addition, the dataset provides a ground 
truth manually labeled by several neuroradiologists, which corresponds to the manual segmentation. Labelling the MR 
images by using the four sequences is a tough task and takes a long time. In this work, it is proposed a light and fast 
deep  learning architecture to deal with the problem at hand automatically.

The main contributions of the work are the following:

- We propose a novel efficient lightweight architecture that outperforms some of the most popular architectures previously applied to biomedical image segmentation, such as U-Net or V-Net. Our model has significantly lower training and computational requirements while achieving comparable performance

- The study shows statistical differences between LGG and HGG, and suggests a potential shift in glioma segmentation strategies to optimize outcomes, especially for HGG tumors, the most aggressive.
  
- We prove that the brain tumor segmentation problem can be effectively approached in a 2-step way by first classifying the type of glioma between HGG and LGG, and then segmenting the MRI.



