# MEG-MRI preprocessing pipeline (2023-09-08, RMCL)

## Summary of MRI preprocessing pipeline
0. Convert DICOM images to NIFTI format: `00_setup_subject.py`
1. Reconstruct cortical surfaces: `cd mri` then `./01_recon_all_batch.sh`
2. Create BEM meshes: `./02_watershed_bem_batch.sh`
3. Coregister MEG and MRI data: 
    ```
    conda activate mne1.4.2
    module load mnelib
    mne coreg
    # save sub-xx_trans.fif in subjects_dir
    ```
## Summary of MEG preprocessing pipeline
0. Convert MEG data to MNE-BIDS format: `00_setup_subject.py`
1. Separate distal sources from biological sources: `01_maxfilter_batch.sh`
2. Suppress artifact and segment continuous data: `02_preprocess_data.py`
    - bandpass filter 0.1-40 Hz (FIR, Hamming window)
    - ica to remove eye-related components (~2-3 componets) (not heart-related components)
    - correct for delay between trigger and display via photodiode
    - segment continuous data into epochs
    - downsample to 250 Hz
    - use `autoreject` to determine amplitude rejection threshold
3. Calculate noise covariance matrices: `03_noise_cov.py`
4. Calculate forward solution: `04_forward_solution.py`
5. Calculate inverse solution: `05_inverse_solution.py`
6. Estimate sources: `06_source_estimation.py`

***

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
