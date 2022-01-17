import logging
from typing import List
from pathlib import Path
import numpy as np
import torch
from numpy import logical_and as l_and, logical_not as l_not
from scipy.spatial.distance import directed_hausdorff

HAUSSDORF = "Haussdorf distance"
DICE = "DICE"
SENS = "Sensitivity"
SPEC = "Specificity"
ACC = "Accuracy"
JACC = "Jaccard index"
PREC = "Precision"
METRICS = [HAUSSDORF, DICE, SENS, SPEC, ACC, JACC, PREC]


def calculate_metrics(
        ground_truth: np.ndarray,
        segmentation: np.ndarray,
        patient: str,
        regions: List[str]
) -> List[dict]:
    """
    This function computes the Jaccard index, Accuracy, Haussdorf, DICE score, Sensitivity, Specificity and Precision.

    True Positive: Predicted presence of tumor and there were tumor in ground truth
    True Negative: Predicted not presence of tumor and there were not tumor in ground truth
    False Positive: Predicted presence of tumor and there were not tumor in ground truth
    False Negative: Predicted not presence of tumor and there were tumor in ground truth

    Params
    ******
        - ground_truth (torch.Tensor): Torch tensor ground truth of size 1*C*Z*Y*X
        - segmentation (torch.Tensor): Torch tensor predicted of size 1*C*Z*Y*X
        - patient (String): The patient ID
        - regions (List[String]): Regions to predict


    Returns
    *******
        - metrics (List[dict]): List of dict {metric:value} where each element of list represent every
                                label (ET, TC, WT)
    """

    assert segmentation.shape == ground_truth.shape, "Predicted segmentation and ground truth do not have the same size"

    metrics_list = []
    for i, region in enumerate([region.upper() for region in regions]):
        metrics = dict(
            patient_id=patient,
            region=region
        )

        # Ground truth and segmentation for region i-th (et, tc, wt)
        gt = ground_truth[i]
        seg = segmentation[i]

        #  cardinalities metrics tp, tn, fp, fn
        tp = float(np.sum(l_and(seg, gt)))
        tn = float(np.sum(l_and(l_not(seg), l_not(gt))))
        fp = float(np.sum(l_and(seg, l_not(gt))))
        fn = float(np.sum(l_and(l_not(seg), gt)))

        # If a region is not present in the ground truth some metrics are not defined
        if np.sum(gt) == 0:
            logging.info(f"{region} not present for {patient}")

        # Computing all metrics
        metrics[HAUSSDORF] = haussdorf_distance(gt, seg)
        metrics[DICE] = dice_score(tp, fp, fn, gt, seg)
        metrics[SENS] = sentitivity(tp, fn)
        metrics[SPEC] = specificity(tn, fp)
        metrics[ACC] = accuracy(tp, tn, fp, fn)
        metrics[JACC] = jaccard_index(tp, fp, fn, gt, seg)
        metrics[PREC] = precision(tp, fp)

        metrics_list.append(metrics)

    return metrics_list


def save_metrics(
        metrics: List[torch.Tensor],
        current_epoch: int,
        regions: List[str],
        save_folder: Path = None
):
    """
    This function is called after every validation epoch to store metrics into .txt file.


    Params:
    *******
        - metrics (torch.nn.Module): model used to compute the segmentation
        - current_epoch (int): number of current epoch
        - classes (List[String]): regions to predict
        - save_folder (Path): path where the model state is saved

    Return:
    *******
        - It does not return anything. However, it generates a .txt file where is stored the results got in the
        validation step. filename = validation_error.txt
    """

    metrics = list(zip(*metrics))
    metrics = [torch.tensor(metric, device="cpu").numpy() for metric in metrics]
    metrics = {key: value for key, value in zip(regions, metrics)}
    print(f"\nEpoch {current_epoch} -> "
          f"Val: {[f'{key.upper()} : {np.nanmean(value):.4f}' for key, value in metrics.items()]} -> "
          f"Average: {np.mean([np.nanmean(value) for key, value in metrics.items()]):.4f}"
          )


    # Saving progress in a file
    with open(f"{save_folder}/validation_error.txt", mode="a") as f:
        print(f"\nEpoch {current_epoch} -> "
              f"Val: {[f'{key.upper()} : {np.nanmean(value):.4f}' for key, value in metrics.items()]} -> "
              f"Average: {np.mean([np.nanmean(value) for key, value in metrics.items()]):.4f}",
              file=f)


def sentitivity(tp: float, fn: float) -> float:
    """
    The sentitivity is intuitively the ability of the classifier to find all tumor voxels.
    """

    if tp == 0:
        sensitivity = np.nan
    else:
        sensitivity = tp / (tp + fn)

    return sensitivity


def specificity(tn: float, fp: float) -> float:
    """
    The specificity is intuitively the ability of the classifier to find all non-tumor voxels.
    """

    spec = tn / (tn + fp)

    return spec


def precision(tp: float, fp: float) -> float:

    if tp == 0:
        prec = np.nan
    else:
        prec = tp / (tp + fp)

    return prec


def accuracy(tp: float, tn: float, fp: float, fn: float) -> float:

    return (tp + tn) / (tp + tn + fp + fn)


def dice_score(tp: float, fp: float, fn: float, gt: np.ndarray, seg: np.ndarray) -> float:

    if np.sum(gt) == 0:
        dice = 1 if np.sum(seg) == 0 else 0
    else:
        dice = 2 * tp / (2 * tp + fp + fn)

    return dice


def jaccard_index(tp: float, fp: float, fn: float, gt: np.ndarray, seg: np.ndarray) -> float:

    if np.sum(gt) == 0:
        jac = 1 if np.sum(seg) == 0 else 0
    else:
        jac = tp / (tp + fp + fn)

    return jac


def haussdorf_distance(gt: np.ndarray, seg: np.ndarray) -> float:

    if np.sum(gt) == 0:
        hd = np.nan
    else:
        hd = directed_hausdorff(np.argwhere(seg), np.argwhere(gt))[0]

    return hd
