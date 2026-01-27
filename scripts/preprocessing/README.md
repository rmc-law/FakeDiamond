# MEG-MRI preprocessing pipeline (2023-09-08, RMCL)
The preprocessing pipeline uses MNE-Python (v1.4.2) to transform raw MEG/EEG sensor data into cortical source estimates. It integrates concurrent 306-channel MEG, 64-channel EEG, and structural MRI.

## Execution order

### MEG-EEG steps
1. `00_setup_subject.py` & `00_organise_logfiles.py`: Standardises the directory structure and prepares participant metadata.
2. `01_maxfilter_batch.sh`: Performs Maxwell filtering (SSS) and head movement compensation for noise reduction.
3. `02_preprocess_data.py`: Handles filtering (0.1–40 Hz), ICA-based artifact rejection, and automated trial rejection via autoreject.
4. `03_noise_cov.py`: Estimates the noise covariance matrix from the pre-stimulus baseline.
5. `04_forward_solution.py`: Constructs a three-layer boundary element model (BEM) and computes the gain matrix.
6. `05_inverse_solution.py`: Assembles the L2-MNE inverse operator.
7. `06_source_estimation.py`: Projects sensor data to a common cortical space (fsaverage) for group analysis.

### MRI steps
1. `00_setup_subject.py`: Converts DICOM images to NIFTI format
2. `01_recon_all_batch.sh`: Reconstructs cortical surfaces
3. `02_watershed_bem_batch.sh`: Creates BEM meshes
4. Coregister MEG and MRI data: 
    ```
    conda activate mne1.4.2
    module load mnelib
    mne coreg
    # save sub-xx_trans.fif in subjects_dir
    ```


## 1. Convert dicom to nifti

After subject BIDS has been set up, we can convert dicom images to nifti for cortical surface reconstruction.
```
conda activate mne1.4.2
cd /imaging/hauk/rl05/fake_diamond/scripts/preprocessing/
python 00_setup_subject.py
```

To check the quality of the MR structural images, use FSLeyes.
```
module load fsl
fsleyes
```
Open the nifti files to view the 3D MR imges. 

> N.B.: Add a single image at a time and do not overlay different images! This can distort the view of the images. 



## 2. Reconstruct cortical surfaces using Freesurfer

After subject dicom images have been converted to nifti format, run this on cluster:
```
cd /imaging/hauk/rl05/fake_diamond/scripts/preprocessing/mri
sbatch 01_recon_all_batch.sh
```

In this setup, the `01_recon_all_batch.sh` script reads the `subject_ids` from my config script, and for each subject ID, it submits a separate job using sbatch and passes the subject ID and main recon_all script path as export variables.

Each individual subject job (`recon_all_job.sh`) runs the main script for a specific subject using the passed subject ID. The %j in the output and error file paths of `recon_all_job.sh` is replaced with the job ID automatically by sbatch.

The `recon_all.sh` script then takes the `subject_id` and runs `recon_all` for that subject.

Note: make sure that these `.sh` scripts are executable:
chmod +x 01_recon_all_batch.sh recon_all_job.sh recon_all.sh

> To check job status: squeue -u <your_username>
