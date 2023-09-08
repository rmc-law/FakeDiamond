# MRI preprocessing pipeline

Three steps to preprocess MRI

For each subject
1. Convert dicom images into NIFTI format -- `convert_dicom2nifti.py`
2. Use NIFTI files to reconstruct cortical surfaces using FreeSurfer -- `batch_recon.sh`
3. Create BEM surfaces using the FreeSurfer watershed algorithm -- `batch_watershed.sh`
