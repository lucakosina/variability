import nilearn
from nilearn.glm.first_level import FirstLevelModel
from nilearn.glm.first_level import make_first_level_design_matrix
from nilearn.plotting import plot_design_matrix
from nilearn.glm import threshold_stats_img
from nilearn.plotting import plot_contrast_matrix, plot_stat_map, show, plot_img
from nilearn.image import mean_img
from nilearn.image import concat_imgs, math_img
from nilearn.plotting import plot_glass_brain
import nibabel as nib
import nilearn.image
from nilearn.input_data import NiftiLabelsMasker, NiftiMasker
from nilearn import plotting
from nilearn.datasets import fetch_atlas_schaefer_2018
from nilearn.datasets import fetch_atlas_harvard_oxford

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from joblib import Parallel, delayed
import os


class lss_variability_analysis:

    def __init__(self):
        pass    

    def load_data(self, folder, subID, run = None, mode = None, condition = None):
        sub_folder = subID + "/"
        run_folder = run + "/"

        # check if folder is empty
        folder_path = os.path.expanduser(folder + sub_folder + "EMOTION_" + run_folder)
        if not os.listdir(folder_path):
            #warnings.warn(f"The folder is empty: {folder_path}")
            raise FileNotFoundError(f"No files found in the folder: {folder_path}")
        
        events_neut = pd.read_csv(folder + sub_folder + "EMOTION_" + run_folder + "EVs/neut.txt", sep="\t", header=None, names=["onset", "duration", "?"])
        events_faces = pd.read_csv(folder + sub_folder + "EMOTION_" + run_folder + "EVs/fear.txt", sep="\t", header=None, names=["onset", "duration", "?"])

        if mode == "blocks":
            events_neut["trial_type"] = [f'neut_{i+1}' for i in range(len(events_neut))]
            events_faces["trial_type"] = [f'face_{i+1}' for i in range(len(events_faces))]
        elif mode == "conditions":
            events_neut["trial_type"] = "neut"
            events_faces["trial_type"] = "face"
        elif mode == "trials":
            #seperate each block into 6 trials of 3s each
            events_neut = events_neut.loc[events_neut.index.repeat(6)].reset_index(drop=True)
            events_faces = events_faces.loc[events_faces.index.repeat(6)].reset_index(drop=True)
            events_neut["duration"] = 3.0
            events_faces["duration"] = 3.0
            events_neut["onset"] = events_neut["onset"] + (events_neut.groupby(events_neut.index // 6).cumcount()) * 3.0
            events_faces["onset"] = events_faces["onset"] + (events_faces.groupby(events_faces.index // 6).cumcount()) * 3.0
            events_neut["trial_type"] = [f'neut_{i+1}' for i in range(len(events_neut))]
            events_faces["trial_type"] = [f'face_{i+1}' for i in range(len(events_faces))]
        else:
            raise ValueError("mode must be 'blocks', 'conditions', or 'trials'")
            
        events = pd.concat([events_neut, events_faces], ignore_index=False)
        events = events.sort_values(by="onset").reset_index(drop=True)

        # cut out distorted end trials
        events = events.iloc[:-4]

        fmri_img = nib.load(folder + sub_folder + "EMOTION_" + run_folder + "tfMRI_EMOTION_" + run + ".nii.gz")
        fmri_data = fmri_img.get_fdata()  
        # smooth image
        fmri_img = nilearn.image.smooth_img(fmri_img, fwhm=6)

        motion_confounds = pd.read_csv(folder + sub_folder + 'EMOTION_' + run + '/Movement_Regressors.txt', delim_whitespace=True, header=None)
        
        motion_confounds.columns = [
            "X", "Y", "Z",        
            "RotX", "RotY", "RotZ",  
            "dX", "dY", "dZ",       
            "dRotX", "dRotY", "dRotZ"
        ]

        return fmri_img, events, motion_confounds

    def make_first_level_model(self, fmri_img, events, confounds):

        t_r = 0.72
        n_scans = fmri_img.shape[-1]  
        frame_times = np.arange(n_scans) * t_r  

        design_matrix = make_first_level_design_matrix(
            frame_times,
            events,
            hrf_model="spm",
            drift_model="cosine",
            high_pass=0.01,
            add_regs=confounds,
            add_reg_names=confounds.columns.tolist()
        )

        first_level_model = FirstLevelModel(t_r=t_r, slice_time_ref=0.5, hrf_model="spm", drift_model="cosine", high_pass=0.01)
        first_level_model = first_level_model.fit(fmri_img, design_matrices=design_matrix)

        design_columns = design_matrix.columns
        # turn design_columns into panadas Index
        design_columns = pd.Index(design_columns)

        return first_level_model, design_matrix, design_columns
    
    def get_beta_maps_ls_s(self, fmri_img, events, confounds):
        """
        Calculate beta maps for each trial using the Least Squares - Single (LSS) method.
        This models each trial separately while collapsing all other trials into a single regressor.
        
        Parameters:
        fmri_img (nib.Nifti1Image): The preprocessed fMRI data.
        events (pd.DataFrame): DataFrame containing event information with columns ['onset', 'duration', 'trial_type'].
        confounds (pd.DataFrame): DataFrame containing confound regressors (e.g., motion parameters).
        
        Returns:
        beta_maps (list of nib.Nifti1Image): List of beta maps for each trial.
        beta_sd_map (nib.Nifti1Image): Voxelwise standard deviation map across all beta maps.
        
        """
                
        t_r = 0.72
        n_scans = fmri_img.shape[-1]
        frame_times = np.arange(n_scans) * t_r

        beta_maps = []

        # iterate over each actual trial in the events DataFrame
        for i, trial in events.iterrows():
            # isolate this trial
            this_trial = trial.to_frame().T.copy()
            this_trial["trial_type"] = f"trial_{i+1}"

            # collapse all other regressors into "other"
            others = events.drop(i).copy()
            others["trial_type"] = "other"

            trial_events_df = pd.concat([this_trial, others], ignore_index=True)

            design_matrix = make_first_level_design_matrix(
                frame_times,
                trial_events_df,
                hrf_model="spm",
                drift_model="cosine",
                high_pass=0.01,
                add_regs=confounds,
                add_reg_names=confounds.columns.tolist()
            )

            first_level_model = FirstLevelModel(
                t_r=t_r,
                slice_time_ref=0.5,
                hrf_model="spm",
                drift_model="cosine",
                high_pass=0.01
            )
            first_level_model = first_level_model.fit(fmri_img, design_matrices=design_matrix)

            # extract beta for this trial only
            eff_map = first_level_model.compute_contrast(f"trial_{i+1}", output_type="effect_size")
            beta_maps.append(eff_map)

        # stack into 4D image and compute voxelwise SD
        beta_4d = concat_imgs(beta_maps)
        beta_sd_map = math_img("np.std(img, axis=-1)", img=beta_4d)

        return beta_maps, beta_sd_map


    def apply_ROI_masks_new(self, img, atlas, region_names, plot = True):
        """
        Apply ROI masks from the given atlas to the input image and extract mean values for specified regions.

        Parameters:
        img (nib.Nifti1Image): The input 3D NIfTI image.
        atlas (nilearn.datasets): The atlas containing ROI masks and labels.
        region_names (list of str): List of region names to filter and extract from the ROI names in the atlas
        plot (bool): Whether to plot the results.

        Returns:
        dict: A dictionary with region names as keys and their corresponding mean values as values.

        """
                
        masker = NiftiLabelsMasker(
            labels_img=atlas.maps,
            labels=atlas.labels,
            standardize=False
        )

        roi_signals = masker.fit_transform(img)
        roi_signals = roi_signals.ravel()

        all_ROI_maps = {}

        for name in region_names:
            print(f"Processing region filter: {name}")

            region_indices = []
            for j, label in enumerate(atlas.labels):
                if name in label:
                    print(f"Found match: {label} at index {j}")
                    region_index = j 
                    region_indices.append(region_index)

            for idx in region_indices:
                if idx == 0:
                    continue
                label_name = atlas.labels[idx] 
                roi_value = roi_signals[idx-1]

                all_ROI_maps[label_name] = roi_value
                print(f"Value for {label_name}: {roi_value}")
            
        if plot:
                plt.figure(figsize=(10, 6))

                for i, (region, values) in enumerate(all_ROI_maps.items()):
                    plt.scatter(np.full_like(values, i + 1), values, alpha=0.6, label=region)
                    mean = np.mean(values)
                    plt.hlines(mean, i + 0.8, i + 1.2, colors='red', linestyles='dashed')
                    std = np.std(values)
                    plt.fill_betweenx([mean - std, mean + std], i + 0.8, i + 1.2, color='red', alpha=0.2)
                    plt.text(i + 1.25, mean, f'Mean: {mean:.2f}\nSD: {std:.2f}', verticalalignment='center')

                plt.legend()
                plt.xlabel('ROIs')
                plt.ylabel('Variability (SD of Beta values)')
                plt.title('Variability across ROIs')
                plt.show()

        return all_ROI_maps
    
    def calculate_interregional_variability(self, all_ROI_maps, plot=False):
        rois = all_ROI_maps.columns
        n = len(rois)

        data = all_ROI_maps.T.values  
        diffs = np.abs(data[:, None, :] - data[None, :, :])
        matrix = diffs.mean(axis=2)
        matrix_df = pd.DataFrame(matrix, index=rois, columns=rois)

        if plot:
            plt.figure(figsize=(12, 10))
            plt.imshow(matrix, cmap="viridis", interpolation="nearest")
            plt.colorbar(label="Absolute Difference in Variability")
            plt.xticks(ticks=np.arange(n), labels=rois, rotation=90)
            plt.yticks(ticks=np.arange(n), labels=rois)
            plt.title("Interregional Variability Differences")
            plt.show()

        return matrix_df




### Main script

# list of subjIDs
subjectIDs = pd.read_csv("/home/luca/HCP_tasks/subjects.txt", header=None).squeeze()
subjectIDs = subjectIDs.tolist()
subIDs = [str(s) for s in subjectIDs]

outliers = [101107, 107725, 114924, 136126, 140824, 154330, 156233, 171532, 181636, 186444, 192237, 200917, 203923, 208428, 211114, 308129, 599671, 756055, 878776, 901038, 943862]

subIDs = [subID for subID in subIDs if int(subID) not in outliers]


var = lss_variability_analysis()

# run for all subjects and save the beta maps

atlas = fetch_atlas_schaefer_2018(n_rois=100, yeo_networks=7, resolution_mm=1, verbose=1)

atlas_sub = fetch_atlas_harvard_oxford('sub-maxprob-thr50-1mm')

masker_schaefer = NiftiLabelsMasker(labels_img=atlas.maps, labels=atlas.labels, standardize=False)
masker_schaefer.fit() 
masker_sub = NiftiLabelsMasker(labels_img=atlas_sub.maps, labels=atlas_sub.labels, standardize=False)
masker_sub.fit()

regions_schaefer = atlas.labels[1:]
regions_sub = ["Amygdala", "Hippocampus", "Putamen", "Thalamus", "Accumbens"]


folder = '../HCP_tasks/'


def process_subject(subID, folder, atlas, atlas_sub, regions_schaefer, regions_sub, condition=None):
    try:
        print(f"Processing subject {subID}...")

        # check if folder already contains file then skip
        folder_path = os.path.expanduser(folder + subID + "/")
        file_name = f"lss_ROI_variability_{subID}_{condition}.csv"
        file_path = os.path.join(folder_path, file_name)

        if os.path.exists(file_path):
            print("Subject already processed, skipping...")
            return None
        else:

            fmri_img, events, confounds = var.load_data(folder, subID, run="LR", mode="trials")
            betas_LR, beta_maps_LR = var.get_beta_maps_ls_s(fmri_img, events, confounds)

            fmri_img, events, confounds = var.load_data(folder, subID, run="RL", mode="trials")
            betas_RL, beta_maps_RL = var.get_beta_maps_ls_s(fmri_img, events, confounds)

            beta_maps_all = nilearn.image.mean_img([beta_maps_LR, beta_maps_RL])
            output_path = folder + subID + "/" + f"lss_beta_sd_map_{subID}_{condition}.nii.gz"
            nib.save(beta_maps_all, output_path)

            ROI_maps_schaefer = var.apply_ROI_masks_new(beta_maps_all, atlas=atlas, region_names=regions_schaefer)
            ROI_maps_sub = var.apply_ROI_masks_new(beta_maps_all, atlas=atlas_sub, region_names=regions_sub)
            all_ROI_maps = {**ROI_maps_schaefer, **ROI_maps_sub}

            all_ROI_maps_df = pd.DataFrame.from_dict(all_ROI_maps, orient='index').T

            all_ROI_maps_df.to_csv(folder + subID + "/" + f"lss_ROI_variability_{subID}_{condition}.csv")

            return subID, beta_maps_all, all_ROI_maps
        
    except FileNotFoundError as e:
        print(f"⚠️ Skipping {subID}: {e}")
        return None
    

results = Parallel(n_jobs=8)(delayed(process_subject)(
    subID, folder, atlas, atlas_sub, regions_schaefer, regions_sub, condition="trials"
) for subID in subIDs)



"""
# calculate the interregional variability for each subject and save the matrices
interregional_variabilities = {}
for subID in subIDs:
    file_path = folder + subID + "/" + f"lss_ROI_variability_{subID}_trials.csv"
    print(file_path)
    if os.path.exists(file_path):
        roi_data = pd.read_csv(file_path)
        diffs_matrix = var.calculate_interregional_variability(roi_data)
        interregional_variabilities[subID] = diffs_matrix
        # save diffs_matrix as npy file
        output_path = folder + subID + "/" + f"lss_interregional_variability_{subID}_14_10_25.csv"
        diffs_matrix.to_csv(output_path)
        print(f"Saved {output_path}")
"""
