import os
import json
import torch
import random
import pprint
import logging
import argparse
import numpy as np
import pandas as pd
from typing import Tuple
from torch.cuda.amp import autocast
from sklearn.metrics import accuracy_score
from SimpleITK import GetImageFromArray, GetArrayFromImage, WriteImage, ReadImage
from ..utils.metrics import METRICS
from ..utils.metrics import calculate_metrics
from ..dataset.BraTS_dataset import recover_initial_resolution
import matplotlib.pyplot as plt
import seaborn as sns


def save_args(args: argparse.Namespace):
    """
    This function saves parsed arguments into config file.


    Parameters
    ----------
    args (dict{arg:value}): Arguments for this run

    """

    config = vars(args).copy()
    del config['save_folder'], config['seg_folder']

    logging.info(f"Execution for configuration:")
    pprint.pprint(config)

    config_file = args.save_folder / "config_file.json"
    with config_file.open("w") as file:
        json.dump(config, file, indent=4)


def stats_tensor(tensor: torch.Tensor, detach: bool = False, decimals: int = 6):
    """
    This function takes as input a tensor and calculates a set of statistic values. For example, maximum,
    minimum, mean, median and standard deviation

    Params:
    *******
        - tensor (torch.Tensor): Tensor over which calculate the stats
        - detach (bool, optional):
        - decimals (int, optional): Number of decimals in the stats

    """
    if detach:
        nd_array = tensor.detach().cpu().numpy()[np.nonzero(tensor.detach().cpu().numpy())]
    else:
        nd_array = tensor.cpu().numpy()[np.nonzero(tensor.cpu().numpy())]
    max_, min_ = np.max(nd_array), np.min(nd_array)
    mean, median = np.mean(nd_array), np.median(nd_array)
    std = np.std(nd_array)

    logging.info(f"Statistics -> "
                 f"Max: {max_:.{decimals}f} | "
                 f"Min: {min_:.{decimals}f} | "
                 f"Mean: {mean:.{decimals}f} | "
                 f"Median: {median:.{decimals}f} | "
                 f"Std: {std:.{decimals}f}")


def model_prediction(model: torch.nn.Module, sequences: torch.tensor, device: str = 'cpu') -> np.ndarray:
    """
    This function takes as input the sequences of a MRI and a model to generate an automatic segmentation.

    Params:
    *******
        - model (torch.nn.Module): Trained model
        - sequences (torch.tensor): Sequences to consider
        - device (str): Device to use i.e. "cpu" or "gpu"
    Return:
    *******
        segmentation (np.array): Automatic segmentation generated by the model

    """
    sequences = sequences.to(device)
    auto_cast_bool = True
    if device == 'cpu':
        auto_cast_bool = False

    with autocast(enabled=auto_cast_bool):
        with torch.no_grad():
            segmentation = model(sequences)
            if type(segmentation) == list:
                segmentation = segmentation[-1]
            segmentation = torch.sigmoid(segmentation[0]).cpu().numpy() > 0.5

    return segmentation


def regions_to_labels(segmentation: np.ndarray, regions: Tuple[str]) -> np.ndarray:
    """
    This function takes a input a segmentation as a 3D np.array and calculates the original labels from the regions

    Params:
    *******
        - segmentation (torch.Tensor): segmentation predicted by the model
        - regions (tuple[str]): regions to assess

    Return:
    *******
        - label_map (np.array): Array which contains all the labels
            - et: enhancing tumor label
            - net: necrotic and non-enhancing tumor core label
            - ed: peritumoral edema label

    """

    if len(regions) == 3:
        et = segmentation[0]
        net = np.logical_and(segmentation[1], np.logical_not(et))
        ed = np.logical_and(segmentation[2], np.logical_not(segmentation[1]))
    elif len(regions) == 1:
        if "et" in regions:
            et = segmentation[0]
        elif "tc" in regions:
            net = segmentation[0]
        elif "wt" in regions:
            ed = segmentation[0]
    else:
        assert "ERROR: The number of regions should be 1 or 3"

    label_map = np.zeros(segmentation[0].shape)
    if "et" in vars():
        label_map[et] = 4
    if "net" in vars():
        label_map[net] = 1
    if "ed" in vars():
        label_map[ed] = 2

    return label_map


def generate_boxplot_metrics(metrics_df: pd.DataFrame, path: str):
    """
    This function generates a boxplot for each metric measured.

    Params:
    *******
        - metrics_df (pd.DataFrame): Dataframe which contains the information
        - path (str): Path where the images are stored

    Return:
    *******
        - It does not return anything, but it store a set of boxplot in the path specified.

    """
    os.mkdir(path)
    for metric in METRICS:
        boxplot = metrics_df.boxplot(column=metric, by="region", figsize=(12, 12), fontsize=20, grid=False).get_figure()
        # boxplot.set(title='')
        sns.despine(left=False, bottom=False)
        plt.xlabel("")
        plt.ylabel(metric, fontsize=20)
        # boxplot.suptitle('')
        plt.savefig(f"{path}{metric}.png")



def generate_segmentations(
        data_loader: torch.utils.data.dataloader.DataLoader,
        model: torch.nn.Module,
        args: argparse.Namespace,
        device="cpu"
):
    """
    This function takes a model and torch DataLoader to generate a segmentation. It also store the sequences and
    ground truth cropped, and the segmentation predicted.

    Params:
    *******
        - data_loader (torch.utils.data.dataloader.DataLoader): torch DataLoader which contains information of
                                                                the patient (id, sequences, gt, ..)
        - model (torch.nn.Module): model used to compute the segmentation
        - writer (SummaryWriter): tensorboard object in which we write some results
        - args (dict{arg:value}): arguments for this run

    Return:
    *******
        It does not return anything. However, it generates several images contained into args.save_folder/metrics and
        some .csv files which store a summary of metrics got by segmentations predicted and results got for each patient
    """

    metrics_list = []
    for _, batch in enumerate(data_loader):
        ref_path = ['./../datasets/BRATS2020/TrainingData/BraTS20_Training_001/BraTS20_Training_001_seg.nii.gz']
        ref_path = ['./../datasets/BRATS2021/TrainingData/BraTS2021_00002/BraTS2021_00002_seg.nii.gz']
        # Getting image attributes
        sequences = batch["sequences"]
        ground_truth = batch["ground_truth"][0].cpu().numpy()
        patient_id = batch["patient_id"][0]
        cropped_indexes = [(item[0].item(), item[1].item()) for item in batch['cropped_indexes']]
        random_indexes = [(item[0].item(), item[1].item()) for item in batch['random_indexes']]
        logging.info(f"Processing patient {patient_id} ...")

        # Predicting segmentation
        segmentation = model_prediction(model=model, sequences=sequences, device=device)
        logging.info(f"Segmentation calculated...")


        # Evaluating ground truth and segmentation
        patient_metric_list = calculate_metrics(ground_truth=ground_truth, segmentation=segmentation,
                                                patient=patient_id, regions=args.regions)
        metrics_list.append(patient_metric_list)
        logging.info(f"Metrics stored...")

        # Recovering initial resolution and transforming regions to labels
        recovered_segmentation = recover_initial_resolution(image=segmentation, cropped_indexes=cropped_indexes, random_indexes=random_indexes)
        recovered_segmentation = regions_to_labels(segmentation=recovered_segmentation, regions=args.regions)

        # Postprocessing
        if args.postprocessing_threshold > 0:
            pixels_dict = count_pixels(recovered_segmentation)
            if pixels_dict[4.0] < args.postprocessing_threshold:
                logging.info(f"This segmentation has less than {args.postprocessing_threshold} pixels in the ET region. Dropping that label.")
                recovered_segmentation[recovered_segmentation == 4.0] = 1.0


        # Storing segmentation using the input resolution
        recovered_segmentation = GetImageFromArray(np.expand_dims(recovered_segmentation, 0), isVector=False)
        ref_image = ReadImage(ref_path)
        recovered_segmentation.CopyInformation(ref_image) # this step is crutial to maintain the orientation
        WriteImage(recovered_segmentation, f"{args.seg_folder}/{patient_id}.nii.gz")
        logging.info(f"Recovering initial dimensions...")

        # Storing segmentation using the resolution chosen
        segmentation = regions_to_labels(segmentation=segmentation, regions=args.regions)
        ground_truth = regions_to_labels(segmentation=ground_truth, regions=args.regions)
        logging.info(f"Converted ROI segmentation and ground truth into labels...")

        # Saving sequences and ground truth cropped, and segmentation predicted
        np.save(f"{args.seg_folder}/{patient_id}_sequences", sequences.cpu().numpy()[0])
        WriteImage(GetImageFromArray(ground_truth.astype(np.int16)), f"{args.seg_folder}/{patient_id}_ground_truth.nii.gz")
        WriteImage(GetImageFromArray(segmentation.astype(np.int16)), f"{args.seg_folder}/{patient_id}_segmentation.nii.gz")
        logging.info(f"Sequences, ground truth and segmentation saved successfully...")

    # Generating .csv which contains all metrics
    df_metrics_val = pd.DataFrame([item for sublist in metrics_list for item in sublist])
    df_metrics_val.to_csv((args.save_folder / 'results_by_patient.csv'), index=False)

    df_melt = pd.melt(df_metrics_val,
                      id_vars=["patient_id", "region"],
                      value_vars=METRICS,
                      var_name="metric_name",
                      value_name="value")
    df_summary = df_melt.groupby(["region", "metric_name"]).describe().reset_index()
    df_summary.to_csv(f"{args.save_folder}/summary_metrics.csv", index=False)
    logging.info(f"\nMetrics summary:\n\n{df_summary}")

    # Generating boxplot figures for each metric
    generate_boxplot_metrics(metrics_df=df_metrics_val, path=f"{args.save_folder}/metrics/")
    logging.info(f"Boxplot figures stored in the path {args.save_folder}/metrics/")


def generate_classification(
        data_loader: torch.utils.data.dataloader.DataLoader,
        model: torch.nn.Module,
        args: argparse.Namespace,
        device="cpu"
):
    """
    This function takes a model and torch DataLoader to generate a segmentation. It also store the sequences and
    ground truth cropped, and the segmentation predicted.

    Params:
    *******
        - data_loader (torch.utils.data.dataloader.DataLoader): torch DataLoader which contains information of
                                                                the patient (id, sequences, gt, ..)
        - model (torch.nn.Module): model used to compute the segmentation
        - writer (SummaryWriter): tensorboard object in which we write some results
        - args (dict{arg:value}): arguments for this run

    Return:
    *******
        It does not return anything. However, it generates several images contained into args.save_folder/metrics and
        some .csv files which store a summary of metrics got by segmentations predicted and results got for each patient
    """
    auto_cast_bool = True
    if device == 'cpu':
        auto_cast_bool = False

    gts = []
    outs = []
    for _, batch in enumerate(data_loader):
        # Getting image attributes
        sequences = batch["sequences"].to(device)
        ground_truth = batch["grade"][0].int().cpu().numpy()
        patient_id = batch["patient_id"][0]
        logging.info(f"Patient {patient_id} processed")

        # Predicting label
        with autocast(enabled=auto_cast_bool):
            output = model(sequences)
        output = torch.sigmoid(output)[0][0].int().cpu().numpy()
        logging.info(f"Label predicted...")

        gts.append(ground_truth)
        outs.append(output)

    logging.info(f"Ground truths: {gts}")
    logging.info(f"Labels predicted: {outs}")
    logging.info(f"Accuracy: {accuracy_score(gts, outs)}")


def init_log(log_name: str):
    """
    This function initializes a log file.

    Params:
    *******
        - log_name (str): log name

    """
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] - [%(levelname)s] - [%(filename)s:%(lineno)s] --- %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_name,
        filemode='a',
        force=True
    )

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger().addHandler(console)


def seed_everything(seed: int):

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.cuda.manual_seed_all(seed)



class AverageMeter(object):
    """Computes and stores the average and current value."""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        logging.info('\t'.join(entries))

    @staticmethod
    def _get_batch_fmtstr(num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def count_pixels(segmentation):

    unique, counts = np.unique(segmentation, return_counts=True)
    pixels_dict = dict(zip(unique, counts))

    if 4.0 not in pixels_dict:
        pixels_dict[4.0] = 0

    return pixels_dict
