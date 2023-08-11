# MEG-MRI preprocessing pipeline (2023-08-11, RMCL)

Summary of MEG-MRI preprocessing pipeline:
0. Convert MEG data to MNE-BIDS format: `0_setup_subject_bids.py`
1. Convert dicom images to nifti format: `1_convert_dicom2nifti.py`
2. Use nifti files to reconstruct various cortical surfaces: `submit_sbatch.sh`

***

## 1. Convert dicom to nifti

After subject BIDS has been set up, we can convert dicom images to nifti for cortical surface reconstruction.
```
conda activate mne1.4.2
cd /imaging/hauk/rl05/fake_diamond/scripts/preprocessing/
python 1_convert_dicom2nifti.py
```

To check the quality of the MR structural images, use FSLeyes.
```
module load fsl
fsleyes
```
Open the nifti files to view the 3D MR imges. 



## 2. Reconstruct cortical surfaces using Freesurfer

After subject dicom images have been converted to nifti format, run this on cluster:
```
cd /imaging/hauk/rl05/fake_diamond/scripts/preprocessing/mri
sbatch submit_sbatch.sh
```

In this setup, the `submit_sbatch.sh` script reads the `subject_ids` from my config script, and for each subject ID, it submits a separate job using sbatch and passes the subject ID and main recon_all script path as export variables.

Each individual subject job (`submit_subject_recon_job.sh`) runs the main script for a specific subject using the passed subject ID. The %j in the output and error file paths of `submit_subject_recon_job.sh` is replaced with the job ID automatically by sbatch.

The `recon_all.sh` script then takes the `subject_id` and runs the `run_recon_all` function for that subject.

Note: make sure that these `.sh` scripts are executable:
chmod +x submit_sbatch.sh submit_subject_recon_job.sh recon_all.sh

> To check job status: squeue -u <your_username>
