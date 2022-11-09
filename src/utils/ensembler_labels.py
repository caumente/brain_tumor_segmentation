import sys
sys.path.append("./../../")
from collections import Counter
from SimpleITK import GetImageFromArray, WriteImage, ReadImage
from src.utils.dataset import load_nii
import numpy as np
import glob


def filter_label(img, label):
    img[img != label] = 0

    return img


def ensemble_label(imgs):

    ensemble = np.zeros(shape=(imgs[0].shape[0], imgs[0].shape[1], imgs[0].shape[2]), dtype=int)

    for i in range(imgs[0].shape[0]):
        for j in range(imgs[0].shape[1]):
            for k in range(imgs[0].shape[2]):

                preds = [imgs[_][i, j, k] for _ in range(len(imgs))]
                # TODO: if the label is 0 and other, use 0 or the other one
                if set(preds) == 1 or set(preds) == 3:
                    ensemble[i, j, k] = int(preds[0])
                else:
                    ensemble[i, j, k] = int(Counter(preds).most_common(1)[0][0])

    return ensemble


path1 = "./../../experiments/NoAugmentation_ShallowUNet_DeepSupervision/segs/"
path2 = "./../../experiments/MultiImage_ShallowUNet_DeepSupervision24/segs/"
path3 = "./../../experiments/SuperDeepSupervision/segs/"

output_path = "./../../experiments/test_ense/segs/"
ref_image = ReadImage("../../datasets/BRATS2020/TrainingData/BraTS20_Training_001/BraTS20_Training_001_seg.nii.gz")

for n, (path_im1, path_im2, path_im3) in enumerate(zip(sorted(glob.glob(f'{path1}*.nii.gz')),
                                                       sorted(glob.glob(f'{path2}*.nii.gz')),
                                                       sorted(glob.glob(f'{path3}*.nii.gz'))
                                                       )
                                                   ):

    # if n != 27:
    #     continue

    patient_id = path_im1.split("_")[-1].split(".")[0]
    print(patient_id)

    labels = []
    for label in [1, 2, 4]:
        im1 = filter_label(load_nii(path_im1), label=label)
        im2 = filter_label(load_nii(path_im2), label=label)
        im3 = filter_label(load_nii(path_im3), label=label)
        #if label == 1:
        #    labels.append(ensemble_label(imgs=(im1, im2)))
        #else:
        labels.append(ensemble_label(imgs=(im1, im2, im3)))

    ensemble = labels[0] + labels[1] + labels[2]
    ensembled_segmentation = GetImageFromArray(ensemble, isVector=False)
    ensembled_segmentation.CopyInformation(ref_image)  # this step is crucial to maintain the orientation
    WriteImage(ensembled_segmentation, f"{output_path}BraTS20_Validation_{patient_id}.nii.gz")






