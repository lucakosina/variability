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


class WM_lss_variability_analysis:

    def __init__(self):
        pass    

    def load_data_PREACT(self, folder, subID):
        sub_folder = "sub-FOR" + subID + "/"

        # check if folder is empty
        folder_path = os.path.expanduser(folder + "wd_fm_var/derivatives/halfpipe/" + sub_folder + "func/")
        if not os.listdir(folder_path):
            #warnings.warn(f"The folder is empty: {folder_path}")
            raise FileNotFoundError(f"No files found in the folder: {folder_path}")
        
        events_pic_neg1 = pd.read_csv(folder + "task_regressors/WM/FOR"+ subID + "_WM_pic_neg_1.txt", sep=r'\s+', header=None, names=["onset", "duration", "?"])
        events_pic_neg6 = pd.read_csv(folder + "task_regressors/WM/FOR"+ subID + "_WM_pic_neg_6.txt", sep=r'\s+', header=None, names=["onset", "duration", "?"])
        events_pic_neut1 = pd.read_csv(folder + "task_regressors/WM/FOR"+ subID + "_WM_pic_neut_1.txt", sep=r'\s+', header=None, names=["onset", "duration", "?"])
        events_pic_neut6 = pd.read_csv(folder + "task_regressors/WM/FOR"+ subID + "_WM_pic_neut_6.txt", sep=r'\s+', header=None, names=["onset", "duration", "?"])
        events_stim = pd.read_csv(folder + "task_regressors/WM/FOR"+ subID + "_WM_stim.txt", sep=r'\s+', header=None, names=["onset", "duration", "?"])
        events_probe = pd.read_csv(folder + "task_regressors/WM/FOR"+ subID + "_WM_probe.txt", sep=r'\s+', header=None, names=["onset", "duration", "?"])
        events_resp = pd.read_csv(folder + "task_regressors/WM/FOR"+ subID + "_WM_resp.txt", sep=r'\s+', header=None, names=["onset", "duration", "?"])

        # get blocks 
        events_pic_neg1["trial_type"] = [f'neg1_{i+1}' for i in range(len(events_pic_neg1))]
        events_pic_neg6["trial_type"] = [f'neg6_{i+1}' for i in range(len(events_pic_neg6))]
        events_pic_neut1["trial_type"] = [f'neut1_{i+1}' for i in range(len(events_pic_neut1))]
        events_pic_neut6["trial_type"] = [f'neut6_{i+1}' for i in range(len(events_pic_neut6))]
        events_stim["trial_type"] = [f'stim_{i+1}' for i in range(len(events_stim))]
        events_probe["trial_type"] = [f'probe_{i+1}' for i in range(len(events_probe))]
        events_resp["trial_type"] = [f'resp_{i+1}' for i in range(len(events_resp))]

        events = pd.concat([events_pic_neut1, events_pic_neg1, events_pic_neg6, events_pic_neut6, events_stim, events_probe, events_resp], ignore_index=False)
        events = events.sort_values(by="onset").reset_index(drop=True)

        # cut distorted end trials
        #events = events.iloc[:-6]

        end_thres = 20
        # cut out trials in the last 20s of the run
        run_duration = events["onset"].max() + events["duration"].max()
        events = events[events["onset"] + events["duration"] <= run_duration - end_thres].reset_index(drop=True)

        fmri_img = nib.load(folder_path + "sub-FOR" + subID + "_task-wm_setting-preproc_bold.nii.gz")
        fmri_data = fmri_img.get_fdata()  
        # smooth image
        fmri_img = nilearn.image.smooth_img(fmri_img, fwhm=6)


        #motion_confounds = pd.read_csv(folder_path + "sub-FOR"+ subID + "_task-WM_setting-preproc_six_motion_reg.txt", sep='\t', header=None)
        motion_confounds = pd.read_csv(folder_path + 
            "sub-FOR"+ subID + "_task-WM_setting-preproc_six_motion_reg.txt",
            delim_whitespace=True,   
            header=None,            
            names=[
                "trans_x", "trans_y", "trans_z",
                "rot_x", "rot_y", "rot_z"
            ]
        )

        return fmri_img, events, motion_confounds
    
    def get_beta_maps_lss(self, fmri_img, events, confounds):
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

        beta_maps = []

        # iterate over each actual trial in the events DataFrame
        for i, trial in events.iterrows():
            if trial['trial_type'].startswith('neg') or trial['trial_type'].startswith('neut'):
                
                # get trial to model
                this_trial = trial.to_frame().T.copy()
                this_trial["trial_type"] = f"trial_{i+1}"

                # collapse all other regressors into "other"
                others = events.drop(i).copy()
                others["trial_type"] = "other"

                trial_events_df = pd.concat([this_trial, others], ignore_index=True)

                first_level_model = FirstLevelModel(
                    t_r=t_r,
                    slice_time_ref=0.5,
                    hrf_model="spm",
                    drift_model="cosine",
                    high_pass=0.01
                )
                
                first_level_model = first_level_model.fit(
                    fmri_img,
                    events=trial_events_df,
                    confounds=confounds
                )

                design_matrix = first_level_model.design_matrices_[0]
                design_columns = design_matrix.columns

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


folder_PREACT = "/Users/lucakosina/Library/Mobile Documents/com~apple~CloudDocs/Uni/MASTER/lab rotations/ritter lab/PREACT/task/"  

results_folder_PREACT = "/Users/lucakosina/Library/Mobile Documents/com~apple~CloudDocs/Uni/MASTER/lab rotations/ritter lab/PREACT_results/"

subIDs = ["14080"]

var = WM_lss_variability_analysis()

# run for all subjects and save the beta maps

atlas = fetch_atlas_schaefer_2018(n_rois=100, yeo_networks=7, resolution_mm=1, verbose=1)

atlas_sub = fetch_atlas_harvard_oxford('sub-maxprob-thr50-1mm')

masker_schaefer = NiftiLabelsMasker(labels_img=atlas.maps, labels=atlas.labels, standardize=False)
masker_schaefer.fit() 
masker_sub = NiftiLabelsMasker(labels_img=atlas_sub.maps, labels=atlas_sub.labels, standardize=False)
masker_sub.fit()

regions_schaefer = atlas.labels[1:]
regions_sub = ["Amygdala", "Hippocampus", "Putamen", "Thalamus", "Accumbens"]


def process_subject(subID, folder, results_folder, atlas, atlas_sub, regions_schaefer, regions_sub):
    try:
        print(f"Processing subject {subID}...")

        # check if folder already contains file then skip
        folder_path = os.path.expanduser(results_folder + subID + "/")
        file_name = f"lss_ROI_variability_{subID}.csv"
        file_path = os.path.join(folder_path, file_name)

        if os.path.exists(file_path):
            print("Subject already processed, skipping...")
            return None
        
        else:
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            fmri_img, events, confounds = var.load_data_PREACT(folder, subID)
            betas, beta_sd_maps = var.get_beta_maps_lss(fmri_img, events, confounds)

            plotting.plot_glass_brain(beta_sd_maps, colorbar=True, title="LSS Variability Map - WM", plot_abs=False, display_mode="ortho",)
            show()

            output_path = results_folder + subID + "/" + f"lss_beta_sd_map_{subID}.nii.gz"
            nib.save(beta_sd_maps, output_path)

            ROI_maps_schaefer = var.apply_ROI_masks_new(beta_sd_maps, atlas=atlas, region_names=regions_schaefer)
            ROI_maps_sub = var.apply_ROI_masks_new(beta_sd_maps, atlas=atlas_sub, region_names=regions_sub)
            all_ROI_maps = {**ROI_maps_schaefer, **ROI_maps_sub}

            all_ROI_maps_df = pd.DataFrame.from_dict(all_ROI_maps, orient='index').T

            all_ROI_maps_df.to_csv(results_folder + subID + "/" + f"lss_ROI_variability_{subID}.csv")


            return subID, beta_sd_maps, all_ROI_maps
        
    except FileNotFoundError as e:
        print(f"⚠️ Skipping {subID}: {e}")
        return None
    

results = Parallel(n_jobs=8)(delayed(process_subject)(
    subID, folder_PREACT, results_folder_PREACT, atlas, atlas_sub, regions_schaefer, regions_sub
) for subID in subIDs)
