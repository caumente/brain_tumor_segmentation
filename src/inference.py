from argparse import Namespace
from torch.utils.data import DataLoader
import json
import shutil
import os
from pathlib import Path
from src.utils.models import load_checkpoint
from src.utils.models import create_model
from src.utils.miscellany import generate_segmentations
from src.utils.miscellany import seed_everything
import torch
from src.dataset.brats import Brats




experiment_name = "./experiments/Experiment_X_debug_20220107_214722__ShallowUNet_12_batch1_sgd_lr0.0001_epochs400"

with open(f"./{experiment_name}/config_file.json", "r") as read_file:
    args = json.load(read_file)
    args = Namespace(**args)

args.pathdata = "./datasets/BRATS2021/DebugData/"
args.save_folder = Path(f"./{experiment_name}/Inference_{experiment_name.split('/')[-1]}")

print(args)



seed_everything(seed=args.seed)
if os.path.exists(args.save_folder):
    shutil.rmtree(args.save_folder)
args.save_folder.mkdir(parents=True, exist_ok=True)

args.seg_folder = args.save_folder / "segs"
if os.path.exists(args.seg_folder):
    shutil.rmtree(args.seg_folder)
args.seg_folder.mkdir(parents=True, exist_ok=True)


# Checking whether a GPU is available or
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
num_gpus = torch.cuda.device_count()

# Implementing the model, turning it from cpu to gpu, and loading parameters
model = create_model(architecture=args.architecture, sequences=args.sequences, regions=args.regions, width=args.width)
load_checkpoint(f'{str(experiment_name)}/model_best.pth.tar', model)


# Loading dataset
# Checking if the path where the images are exist
pathdata = Path(args.pathdata).resolve()
assert pathdata.exists(), f"Path '{pathdata}' it doesn't exist"
pathdata = sorted([x for x in pathdata.iterdir() if x.is_dir()])
dataset = Brats(patients_path=pathdata,
                sequences=args.sequences,
                has_ground_truth=False,
                regions=args.regions,
                normalization=args.normalization,
                low_percentile=args.low_norm,
                high_percentile=args.high_norm,
                crop_or_pad=args.crop_or_pad,
                fit_boundaries=args.fit_boundaries,
                inverse_seq=args.inverse_seq,
                debug_mode=False)
print(f"Size of dataset: {len(dataset)}")
data_loader = DataLoader(dataset, batch_size=1)


# Generating segmentations
generate_segmentations(data_loader, model, args)









