import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

from src.utils.miscellany import count_pixels
from src.utils.dataset import load_nii
from src.utils.dataset import cleaning_outlier_voxels
from src.utils.dataset import min_max_scaler


def read_sequences(patient_id, cohort, year, t1=True, t1ce=True, t2=True, flair=True, seg=False, clipping=False,
                   normalize=False):
    if year == '21':
        path_images = f'./datasets/BRATS20{year}/{cohort}Data/BraTS20{year}_{cohort}_{patient_id}/BraTS20{year}_' \
                      f'{cohort}_{patient_id}'
    else:
        path_images = f'./datasets/BRATS20{year}/{cohort}Data/BraTS{year}_{cohort}_{patient_id}/BraTS{year}_' \
                      f'{cohort}_{patient_id}'

    if t1:
        t1 = load_nii(Path(f'{path_images}_t1.nii.gz'))
    if t1ce:
        t1ce = load_nii(Path(f'{path_images}_t1ce.nii.gz'))
    if t2:
        t2 = load_nii(Path(f'{path_images}_t2.nii.gz'))
    if flair:
        flair = load_nii(Path(f'{path_images}_flair.nii.gz'))
    if seg:
        seg = load_nii(Path(f'{path_images}_seg.nii.gz'))

    # logic to return only the sequences requested
    seqs = np.array(['t1', 't1ce', 't2', 'flair', 'seg'])

    mask = []
    for s in seqs:
        mask.append(f'{s}' in vars())
    mask = np.array(mask, dtype=bool)

    return_seq = []
    for k in seqs[mask]:
        return_seq.append(vars()[str(k)])

    if clipping:
        return_seq = [cleaning_outlier_voxels(s) for s in return_seq]
    if normalize:
        return_seq = [min_max_scaler(s) for s in return_seq]

    return return_seq


def plot_patient_mri(sequences, labels, _slice, segmentation=None, save_path=None):
    label_name = {
        0: 'Background',
        1: 'Necrotic & non-Enhancing\n tumor core',
        2: 'Peritumoral edema',
        4: 'Enhancing tumor'
    }
    if segmentation is not None:
        fig, ax = plt.subplots(2, 3, figsize=(16, 8))
    else:
        fig, ax = plt.subplots(2, 2, figsize=(12, 8))

    fig.suptitle('MRI sequences', fontsize=30)

    # turn off all axis
    [axi.set_axis_off() for axi in ax.ravel()]

    sequences = np.array([[sequences[0], sequences[1]], [sequences[2], sequences[3]]])
    names = np.array([[labels[0], labels[1]], [labels[2], labels[3]]])

    # Plotting all sequences
    for i in range(2):
        for j in range(2):
            img = ax[i, j].imshow(sequences[i, j][_slice, :, :], cmap='gray')
            plt.colorbar(img, ax=ax[i, j])
            ax[i, j].set_title(names[i, j], fontsize=16)

    if segmentation is not None:
        # Plotting segmentation
        seg_axis = fig.add_subplot(2, 3, 3)
        img = seg_axis.imshow(segmentation[_slice, :, :], cmap='gray')
        seg_axis.set_axis_off()
        seg_axis.set_title('Segmentation', fontsize=16)
        # Legend
        values = np.unique(segmentation[_slice, :, :].ravel())
        colors = [img.cmap(img.norm(value)) for value in values]
        patches = [mpatches.Patch(color=colors[n], label="{l}".format(l=label_name[key])) for n, key in
                   enumerate(values)]
        seg_axis.legend(handles=patches,
                        bbox_to_anchor=(.5, -.7),
                        loc=8,
                        shadow=True,
                        borderpad=1,
                        prop={"size": 12})
        # Add edge color to each element in the legend
        leg = seg_axis.get_legend()
        for _ in range(len(values)):
            leg.legendHandles[_].set_edgecolor('black')

    if save_path is not None:
        plt.savefig(f'{save_path}.jpeg')
    else:
        plt.show()


def plot_results_patient(t1, t1ce, t2, flair, seg, pred, _slice, save_path=None):
    label_name = {
        0: 'Background',
        1: 'Necrotic & non-Enhancing\ntumor core',
        2: 'Peritumoral edema',
        4: 'Enhancing tumor'
    }

    fig, ax = plt.subplots(2, 3, figsize=(16, 8))
    fig.suptitle('MRI sequences, segmentation, and prediction', fontsize=30)

    # turn off all axis
    [axi.set_axis_off() for axi in ax.ravel()]

    sequences = np.array([[t1, t1ce, seg], [t2, flair, pred]])
    names = np.array([['T1', 'T1Gd', 'Segmentation'], ['T2', 'FLAIR', 'Prediction']])

    # Plotting all sequences
    for i in range(2):
        for j in range(3):
            img = ax[i, j].imshow(sequences[i, j][_slice, :, :], cmap='gray')
            if j < 2:
                plt.colorbar(img, ax=ax[i, j])
            ax[i, j].set_title(names[i, j], fontsize=16)
    # Legend
    values = np.unique(seg[_slice, :, :].ravel())
    colors = [img.cmap(img.norm(value)) for value in values]
    patches = [mpatches.Patch(color=colors[n], label="{l}".format(l=label_name[key])) for n, key in enumerate(values)]
    ax[i, j].legend(handles=patches, bbox_to_anchor=(1.1, 1.3), loc=2, shadow=True, borderpad=1, prop={"size": 14})
    # Add edge color to each element in the legend
    leg = ax[i, j].get_legend()
    for l in range(len(values)):
        leg.legendHandles[l].set_edgecolor('black')

    plt.subplots_adjust(wspace=-0.1, hspace=0.15)

    if save_path is not None:
        plt.savefig(f'{save_path}.jpeg')
    else:
        plt.show()


def plot_histograms(t1, t1ce, t2, flair, bins=100, sharex=False, sharey=False, save_path=None):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=sharex, sharey=sharey)
    fig.suptitle('Intensity histograms', fontsize=30)

    # Plotting histograms
    sns.histplot(t1[t1 > 0].ravel(), kde=True, bins=bins, ax=axes[0, 0], color='salmon').set_title('T1', fontsize=16)
    sns.histplot(t1ce[t1ce > 0].ravel(), kde=True, bins=bins, ax=axes[0, 1], color='#EBD70F').set_title('T1Gd',
                                                                                                        fontsize=16)
    sns.histplot(t2[t2 > 0].ravel(), kde=True, bins=bins, ax=axes[1, 0]).set_title('T2', fontsize=16)
    sns.histplot(flair[flair > 0].ravel(), kde=True, bins=bins, ax=axes[1, 1], color='#6CD36E').set_title('FLAIR',
                                                                                                          fontsize=16)

    # Hide axis
    sns.despine(left=False, bottom=False)

    if save_path is not None:
        plt.savefig(f'{save_path}.jpeg')
    else:
        plt.show()


def plot_histograms_boxplots(t1, t1ce, t2, flair, bins=100, save_path=None):
    fig, axes = plt.subplots(4, 2, figsize=(16, 12), gridspec_kw={'height_ratios': [4, .8, 4, .8]})
    # fig, axes = plt.subplots(2, 2, figsize=(16,10))

    fig.suptitle('Intensity histograms', fontsize=30)

    flierprops = dict(markersize=2)
    # Plotting histograms
    sns.histplot(t1[t1 > 0].ravel(), kde=True, bins=bins, ax=axes[0, 0], color='salmon').set_title('T1', fontsize=16)
    sns.boxplot(x=t1[t1 > 0].ravel(), ax=axes[1, 0], color='salmon', flierprops=flierprops).set(xticklabels=[])

    sns.histplot(t1ce[t1ce > 0].ravel(), kde=True, bins=bins, ax=axes[0, 1], color='#EBD70F').set_title('T1Gd',
                                                                                                        fontsize=16)
    sns.boxplot(x=t1ce[t1ce > 0].ravel(), ax=axes[1, 1], color='#EBD70F', flierprops=flierprops).set(xticklabels=[])

    sns.histplot(t2[t2 > 0].ravel(), kde=True, bins=bins, ax=axes[2, 0]).set_title('T2', fontsize=16)
    sns.boxplot(x=t2[t2 > 0].ravel(), ax=axes[3, 0], flierprops=flierprops).set(xticklabels=[])

    sns.histplot(flair[flair > 0].ravel(), kde=True, bins=bins, ax=axes[2, 1], color='#6CD36E').set_title('FLAIR',
                                                                                                          fontsize=16)
    sns.boxplot(x=flair[flair > 0].ravel(), ax=axes[3, 1], color='#6CD36E', flierprops=flierprops).set(xticklabels=[])

    # Hide axis
    sns.despine(left=False, bottom=False)
    for ax in [axes[1, 0], axes[1, 1], axes[3, 0], axes[3, 1]]:
        sns.despine(left=True, bottom=True, ax=ax)
        ax.tick_params(bottom=False, left=False)

    for ax in [axes[0, 0], axes[0, 1], axes[2, 0], axes[2, 1]]:
        ax.set(ylabel='')

    # Horizontal space between subplots
    # plt.subplots_adjust(hspace=0.5)

    if save_path is not None:
        plt.savefig(f'{save_path}.jpeg')
    else:
        plt.show()


def plot_barplot_segmentation(segmentation, save_path=None):
    my_dict = count_pixels(segmentation[segmentation > 0])
    my_dict_label = my_dict.copy()
    my_dict_regions = my_dict.copy()

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # Dict of labels
    my_dict_label['Necrotic & non-Enhancing\n tumor core'] = my_dict_label.pop(1)
    my_dict_label['Peritumoral edema'] = my_dict_label.pop(2)
    try:
        my_dict_label['Enhancing tumor'] = my_dict_label.pop(4)
    except:
        my_dict_label['Enhancing tumor'] = 0
        print("No ET")

    sns.barplot(x=list(my_dict_label.keys()),
                y=list(my_dict_label.values()),
                palette='Blues_d',
                ax=axes[0]).set_title('Number of pixels by label\n\n', fontsize=20)
    for i, v in enumerate(my_dict_label.values()):
        axes[0].text(i - .12, v + 200, str(v), color='black', fontsize=16)

    # Dict of regions
    try:
        my_dict_regions['Whole tumor'] = my_dict_regions[1] + my_dict_regions[2] + my_dict_regions[4]
    except:
        my_dict_regions['Whole tumor'] = my_dict_regions[1] + my_dict_regions[2]
    try:
        my_dict_regions['Tumor core'] = my_dict_regions[1] + my_dict_regions[4]
    except:
        my_dict_regions['Tumor core'] = my_dict_regions[1]
    try:
        my_dict_regions['Enhancing tumor'] = my_dict_regions.pop(4)
    except:
        my_dict_regions['Enhancing tumor'] = 0

    del my_dict_regions[1], my_dict_regions[2]

    sns.barplot(x=list(my_dict_regions.keys()),
                y=list(my_dict_regions.values()),
                palette='Greens_d',
                ax=axes[1]).set_title('Number of pixels by region\n\n', fontsize=20)
    for i, v in enumerate(my_dict_regions.values()):
        axes[1].text(i - .12, v + 500, str(v), color='black', fontsize=16)

    plt.subplots_adjust(wspace=.3)
    sns.despine(left=False, bottom=False)

    if save_path is not None:
        plt.savefig(f'{save_path}.jpeg')
    else:
        plt.show()


def plot_patient_mri_masked(sequences, labels, _slice, segmentation, alpha=.7, save_path=None):
    label_name = {
        0: 'Background',
        1: 'Necrotic & non-Enhancing\n tumor core',
        2: 'Peritumoral edema',
        4: 'Enhancing tumor'
    }

    seg_masked = np.ma.masked_where(segmentation == 0, segmentation)

    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('MRI sequences', fontsize=30)

    # turn off all axis
    [axi.set_axis_off() for axi in ax.ravel()]

    sequences = np.array([[sequences[0], sequences[1]], [sequences[2], sequences[3]]])
    names = np.array([[labels[0], labels[1]], [labels[2], labels[3]]])

    # Plotting all sequences
    for i in range(2):
        for j in range(2):
            img = ax[i, j].imshow(sequences[i, j][_slice, :, :], cmap='gray')
            mask = ax[i, j].imshow(seg_masked[_slice, :, :], cmap='viridis', interpolation='none', vmin=0, alpha=alpha)
            plt.colorbar(img, ax=ax[i, j])
            ax[i, j].set_title(names[i, j], fontsize=16)

    # Legend
    values = np.unique(segmentation[_slice, :, :].ravel())
    colors = [mask.cmap(mask.norm(value)) for value in values]
    patches = [mpatches.Patch(color=colors[n], label="{l}".format(l=label_name[key])) for n, key in enumerate(values)]
    ax[1, 1].legend(handles=patches,
                    bbox_to_anchor=(2, 1.5),
                    loc=8,
                    shadow=True,
                    borderpad=1,
                    prop={"size": 12})

    if save_path is not None:
        plt.savefig(f'{save_path}.jpeg')
    else:
        plt.show()

