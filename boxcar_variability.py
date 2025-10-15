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
import os
from joblib import Parallel, delayed


class var_block:

    def __init__(self):
        pass

    def load_data_PREACT(self, folder, subID, task):
        sub_folder = "sub-FOR" + subID + "/"

        # check if folder is empty
        folder_path = os.path.expanduser(folder + "wd_fm_var/derivatives/halfpipe/" + sub_folder + "func/")
        if not os.listdir(folder_path):
            raise FileNotFoundError(f"No files found in the folder: {folder_path}")
        
        events_faces = pd.read_csv(folder + "task_regressors/" + task + "/FOR"+ subID + "_FM_faces_blocks.txt", sep=r'\s+', header=None, names=["onset", "duration", "?"])
        events_shapes = pd.read_csv(folder + "task_regressors/" + task + "/FOR"+ subID + "_FM_shapes_blocks.txt", sep=r'\s+', header=None, names=["onset", "duration", "?"])

        # get blocks 
        events_shapes["trial_type"] = [f'neut_{i+1}' for i in range(len(events_shapes))]
        events_faces["trial_type"] = [f'face_{i+1}' for i in range(len(events_faces))]
            
        events = pd.concat([events_shapes, events_faces], ignore_index=False)
        events = events.sort_values(by="onset").reset_index(drop=True)

        # potentially cut distorted end trials
        #events = events.iloc[:-2]

        fmri_img = nib.load(folder_path + "sub-FOR" + subID + "_task-fm_setting-preproc_bold.nii.gz")
        fmri_data = fmri_img.get_fdata()  
        # smooth image
        fmri_img = nilearn.image.smooth_img(fmri_img, fwhm=6)


        motion_confounds = pd.read_csv(folder_path + "sub-FOR"+ subID + "_task-fm_setting-preproc_desc-confounds_regressors.tsv", sep='\t', header=None)
        motion_confounds.columns = pd.read_csv(folder_path + "sub-FOR"+ subID + "_task-fm_setting-preproc_desc-confounds_regressors.tsv", sep='\t', nrows=0).columns
        motion_confounds = motion_confounds.iloc[1:]
        motion_confounds = motion_confounds.drop(columns=[
            col for col in motion_confounds.columns if col not in [
            "rot_x", "rot_y", "rot_z",  
            "trans_x", "trans_y", "trans_z", 
            "trans_x_derivative1", "trans_y_derivative1", "trans_z_derivative1",      
            "rot_x_derivative1", "rot_y_derivative1", "rot_z_derivative1"
            ]
        ])

        # seperate images into blocks based on events
        block_imgs_face = []
        block_imgs_shape = []
        for i, row in events.iterrows():
            onset, duration = int(row.onset), int(row.duration)
            if onset + duration > fmri_data.shape[-1]:
                continue
            block = nilearn.image.index_img(fmri_img, slice(onset, onset+duration))
            if "face" in row.trial_type:
                if i == 0:
                    block_imgs_face = [block]
                else:
                    block_imgs_face.append(block)
            else:
                if i == 0:
                    block_imgs_shape = [block]
                else:
                    block_imgs_shape.append(block)
        return fmri_img, block_imgs_face, block_imgs_shape, events_faces, events_shapes, motion_confounds
    

    def load_data_HCP(self, folder, subID, run = None, condition_based = False):
        sub_folder = subID + "/"
        run_folder = run + "/"

        # check if folder is empty
        folder_path = os.path.expanduser(folder + sub_folder + "EMOTION_" + run_folder)
        if not os.listdir(folder_path):
            raise FileNotFoundError(f"No files found in the folder: {folder_path}")
        
        events_neut = pd.read_csv(folder + sub_folder + "EMOTION_" + run_folder + "EVs/neut.txt", sep="\t", header=None, names=["onset", "duration", "?"])
        events_faces = pd.read_csv(folder + sub_folder + "EMOTION_" + run_folder + "EVs/fear.txt", sep="\t", header=None, names=["onset", "duration", "?"])

        # get blocks 
        events_neut["trial_type"] = [f'neut_{i+1}' for i in range(len(events_neut))]
        events_faces["trial_type"] = [f'face_{i+1}' for i in range(len(events_faces))]
            
        events = pd.concat([events_neut, events_faces], ignore_index=False)
        events = events.sort_values(by="onset").reset_index(drop=True)

        # cut distorted end trials for each shapes and faces
        #events = events.iloc[:-2]

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

        # seperate images into blocks based on events
        block_imgs_face = []
        block_imgs_shape = []
        for i, row in events.iterrows():
            onset, duration = int(row.onset), int(row.duration)
            if onset + duration > fmri_data.shape[-1]:
                continue
            block = nilearn.image.index_img(fmri_img, slice(onset, onset+duration))
            if "face" in row.trial_type:
                if i == 0:
                    block_imgs_face = [block]
                else:
                    block_imgs_face.append(block)
            else:
                if i == 0:
                    block_imgs_shape = [block]
                else:
                    block_imgs_shape.append(block)
        return fmri_img, block_imgs_face, block_imgs_shape, events, motion_confounds
    

    def block_merge(self, block_imgs):
        """
        Scale each block by its global mean and concatenate them into a single 4D image.

        Parameters:
        block_imgs (list of nib.Nifti1Image): List of 4D NIfTI images, each representing a block.

        Returns:
        nib.Nifti1Image: A single 4D NIfTI image with all blocks concatenated and scaled.

        """

        all_blocks = []
        for block in block_imgs:
            data = block.get_fdata()
            affine = block.affine
            header = block.header

            # scale by global mean
            global_mean = np.mean(data)
            data_scaled = 100 * data / global_mean

            # back into NIfTI
            block_norm = nib.Nifti1Image(data_scaled, affine, header)
            all_blocks.append(block_norm)

        return concat_imgs(all_blocks)

    def detrend(self, blocks):
        """
        Substract the voxel-wise mean from each block after scaling by the global mean.

        Parameters:
        blocks (list of nib.Nifti1Image or nib.Nifti1Image): List of 4D NIfTI images or a single 4D NIfTI image.

        Returns:
        nib.Nifti1Image: A single 4D NIfTI image with voxel-wise mean subtracted and scaled.

        """

        if isinstance(blocks, nib.Nifti1Image):
            blocks = [blocks]
        elif not isinstance(blocks, (list, tuple)):
            raise TypeError("blocks must be a Nifti1Image or a list/tuple of Nifti1Image")

        block_data_list = [blk.get_fdata(dtype=np.float32) for blk in blocks]

        spatial_shape = block_data_list[0].shape[:3]
        if any(b.shape[:3] != spatial_shape for b in block_data_list):
            raise ValueError("All blocks must have identical spatial dimensions")

        num_vox = int(np.prod(spatial_shape))
        num_scan = sum(b.shape[3] for b in block_data_list)

        result_img = np.zeros((num_scan, num_vox), dtype=np.float32)

        start = 0
        for block_data in block_data_list:
            n_scans = block_data.shape[3]

            # shape -> (n_scans, num_vox)
            block_2d = block_data.reshape(num_vox, n_scans).T  

            # scale block by global mean (100 * block / mean(mean(block)))
            mean_all = block_2d.mean()
            if mean_all == 0:
                raise ValueError("Block has zero global mean - check your data.")
            block_scaled = (100.0 * block_2d) / mean_all  

            # voxel-wise mean and subtract 
            block_mean = block_scaled.mean(axis=0)  
            nonzero_vox = block_mean != 0

            result_img[start:start + n_scans, nonzero_vox] = (
                block_scaled[:, nonzero_vox] - block_mean[nonzero_vox]
            )

            start += n_scans
        result_4d = result_img.T.reshape((*spatial_shape, num_scan))  # shape (X, Y, Z, time)

        # create new NIfTI 
        result_nifti = nib.Nifti1Image(result_4d, affine=blocks[0].affine, header=blocks[0].header)

        return result_nifti

    def shared_var(self, img, metric = None):
        """
        Calculate voxel-wise variability across trials using specified metric.

        Parameters:
        img (nib.Nifti1Image): A 4D NIfTI image with shape (X, Y, Z, trials).
        metric (str): The variability metric to use. 
                    Options are "Var" (variance), "SD" (standard deviation), or "MSSD" (mean squared successive difference).

        Returns:
        nib.Nifti1Image: A 3D NIfTI image with voxel-wise variability.

        """
        
        if metric == "Var":
            return nilearn.image.math_img("np.var(img, axis=-1)", img=img)
        elif metric == "SD":
            return nilearn.image.math_img("np.std(img, axis=-1)", img=img)
        elif metric == "MSSD":
            n_trials = img.shape[-1]
            diff_img = nilearn.image.math_img("np.diff(img, axis=-1)", img=img)
            expr = f"np.sum(diff_img ** 2, axis=-1) / ({n_trials} - 1)"
            return nilearn.image.math_img(expr, diff_img=diff_img)


    def apply_ROI_masks_new(self, img, atlas, region_names, plot = False):
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
                plt.ylabel('Variability (SD of Beta values) - SD')
                plt.title('Variability across ROIs - SD')
                plt.show()

        return all_ROI_maps

    def calculate_interregional_variability(self, all_ROI_maps, plot=False):
        """
        Calculate interregional variability differences between ROIs.
        """

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
        

## main analysis 


folder_PREACT = "../PREACT/task/"  

subIDs = ["11010"]


var = var_block()


atlas = fetch_atlas_schaefer_2018(n_rois=100, yeo_networks=7, resolution_mm=1, verbose=1)

atlas_sub = fetch_atlas_harvard_oxford('sub-maxprob-thr50-1mm')

masker_schaefer = NiftiLabelsMasker(labels_img=atlas.maps, labels=atlas.labels, standardize=False)
masker_schaefer.fit() 
masker_sub = NiftiLabelsMasker(labels_img=atlas_sub.maps, labels=atlas_sub.labels, standardize=False)
masker_sub.fit()

regions_schaefer = atlas.labels[1:]
regions_sub = ["Amygdala", "Hippocampus", "Putamen", "Thalamus", "Accumbens"]

results_folder_PREACT = "../PREACT_results/"


def process_subject(subID, folder, results_folder, metric, atlas, atlas_sub, regions_schaefer, regions_sub):
    try:
        print(f"Processing subject {subID}...")

        # check if folder already contains file then skip
        folder_path = os.path.expanduser(results_folder + subID + "/")
        file_name = f"boxcar_ROI_variability_shapes_{subID}.csv"
        file_path = os.path.join(folder_path, file_name)

        if os.path.exists(file_path):
            print("Subject already processed, skipping...")
            return None
        
        else:

            # create results folder for subject
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            fmri_img, block_imgs_face, block_imgs_shape, events_faces, events_shape, motion_confounds = var.load_data_PREACT(folder, subID, task="FM")

            # apply Boxcar Model to data
            cond_blocks_face = var.block_merge(block_imgs_face)
            cond_blocks_shape = var.block_merge(block_imgs_shape)
            if metric in ["Var", "SD"]:
                img_data_face = var.detrend(cond_blocks_face)
                img_data_shape = var.detrend(cond_blocks_shape)
            else:
                img_data_face = cond_blocks_face 
                img_data_shape = cond_blocks_shape 
            result_img_face = var.shared_var(img_data_face, metric=metric)
            result_img_shape = var.shared_var(img_data_shape, metric=metric)

            # plot & save results
            plotting.plot_glass_brain(result_img_face, colorbar=True, title="Boxcar Variability Map - Faces", plot_abs=False, display_mode="ortho",)
            show()
            plotting.plot_glass_brain(result_img_shape, colorbar=True, title="Boxcar Variability Map - Shapes", plot_abs=False, display_mode="ortho",)
            show()
            
            output_path_faces = results_folder + subID + "/" + f"boxcar_beta_sd_map_faces_{subID}.nii.gz"
            nib.save(result_img_face, output_path_faces)
            print(f"Saved {output_path_faces}")

            output_path_shapes = results_folder + subID + "/" + f"boxcar_beta_sd_map_shapes_{subID}.nii.gz"
            nib.save(result_img_shape, output_path_shapes)


            # apply ROI masks
            block_var_schaefer_face = var.apply_ROI_masks_new(result_img_face, atlas, regions_schaefer)
            block_var_sub_face = var.apply_ROI_masks_new(result_img_face, atlas_sub, regions_sub)
            boxcar_ROIs_face = {**block_var_schaefer_face, **block_var_sub_face}    
            block_var_schaefer_shape = var.apply_ROI_masks_new(result_img_shape, atlas, regions_schaefer)
            block_var_sub_shape = var.apply_ROI_masks_new(result_img_shape, atlas_sub, regions_sub)
            boxcar_ROIs_shape = {**block_var_schaefer_shape, **block_var_sub_shape}

            # save ROI results

            boxcar_ROIs_df_face = pd.DataFrame.from_dict(boxcar_ROIs_face, orient='index').T
            boxcar_ROIs_df_face.to_csv(results_folder + subID + "/" + f"boxcar_ROI_variability_faces_{subID}.csv")
            boxcar_ROIs_df_shape = pd.DataFrame.from_dict(boxcar_ROIs_shape, orient='index').T
            boxcar_ROIs_df_shape.to_csv(results_folder + subID + "/" + f"boxcar_ROI_variability_shapes_{subID}.csv")

            return subID, result_img_face, result_img_shape, boxcar_ROIs_face, boxcar_ROIs_shape
                                                                                
    except FileNotFoundError as e:
        print(f"⚠️ Skipping {subID}: {e}")
        return None


results = Parallel(n_jobs=8)(delayed(process_subject)(
    subID, folder_PREACT, results_folder_PREACT, "SD", atlas, atlas_sub, regions_schaefer, regions_sub, condition_based=True
) for subID in subIDs)
