# src/data_utils.py

import os
import zipfile
import pydicom as dicom
import numpy as np
import copy
import torch
from torch.utils.data import Dataset, DataLoader
import re
from sklearn.model_selection import train_test_split


def sorted_alphanumeric(data):
    """
    Sorts a list in alphanumeric order.
    """
    def convert(text): return int(text) if text.isdigit() else text.lower()
    def alphanum_key(key): return [convert(c)
                                   for c in re.split('([0-9]+)', key)]
    return sorted(data, key=alphanum_key)


def is_valid_dicom(file_path):
    """
    Checks if a file is a valid DICOM file by verifying the 'DICM' prefix.
    """
    try:
        with open(file_path, 'rb') as f:
            f.seek(128)
            magic = f.read(4)
            return magic == b'DICM'
    except:
        return False


def window_image(patient_data, organ, w_width, w_length, rescale=True):
    """
    Applies windowing to the patient's image for a specific organ.
    """
    feature_dict = patient_data['features']
    img = copy.deepcopy(patient_data['img'])  # Deep copy

    # Apply rescale slope and intercept for HU conversion
    slope = feature_dict['RescaleSlope']
    intercept = feature_dict['RescaleIntercept']
    img = (img * slope + intercept)

    # Define window
    img_min = w_length - w_width / 2
    img_max = w_length + w_width / 2

    # Clamp intensities
    img[img < img_min] = img_min
    img[img > img_max] = img_max

    if rescale:
        img = (img - img_min) / (img_max - img_min) * 255.0
        img = img.astype(np.uint8)

    return img


class DicomDataset(Dataset):
    """
    Custom PyTorch Dataset for loading DICOM images and corresponding masks.
    Implements lazy loading by referencing file paths.
    """

    def __init__(self, patient_data_list, hu_window, transform=None, mask_organ='livertumor'):
        """
        Initializes the dataset by storing file paths for images and masks.

        Parameters:
        patient_data_list (list): List of patient data dictionaries.
        hu_window (dict): Dictionary containing windowing parameters.
        transform (callable, optional): Optional transform to be applied on a sample.
        mask_organ (str): The organ mask to use for tumor segmentation.
        """
        self.sample_info = []  # List to hold tuples of (image_path, mask_path)
        self.transform = transform
        self.hu_window = hu_window
        self.mask_organ = mask_organ

        for patient in patient_data_list:
            if patient is None:
                continue
            img_dir = patient['img_dir']
            mask_dir = patient['mask_dirs'].get(mask_organ, None)
            if mask_dir is None:
                print(
                    f"No {mask_organ} can be found with patient {patient['patient_id']}.")

            img_files = sorted_alphanumeric([img for img in os.listdir(img_dir) if 'Zone' not in img])
        		
            mask_files = sorted_alphanumeric(os.listdir(
                mask_dir)) if mask_dir and os.path.exists(mask_dir) else []

            # Ensure that the number of image slices matches the number of mask slices
            if len(img_files) != len(mask_files):
                print(
                    f"Warning: Number of image slices ({len(img_files)}) and mask slices ({len(mask_files)}) do not match for patient {patient['patient_id']}. Skipping this patient.")
                continue

            for img_file, mask_file in zip(img_files, mask_files):
                img_path = os.path.join(img_dir, img_file)
                mask_path = os.path.join(mask_dir, mask_file)
                if is_valid_dicom(img_path) and is_valid_dicom(mask_path):
                    self.sample_info.append((img_path, mask_path))
                else:
                    print(
                        f"Skipping non-DICOM files: {img_path} or {mask_path}")

    def __len__(self):
        return len(self.sample_info)

    def __getitem__(self, idx):
        """
        Retrieves the image and mask at the specified index by loading them on-the-fly.

        Parameters:
        idx (int): Index of the sample to retrieve.

        Returns:
        dict: Dictionary containing the image and mask tensors.
        """
        img_path, mask_path = self.sample_info[idx]

        # Read DICOM files
        try:
            img_ds = dicom.dcmread(img_path, force=True)
            mask_ds = dicom.dcmread(mask_path, force=True)

            img = img_ds.pixel_array.astype(np.float32)
            mask = mask_ds.pixel_array.astype(np.float32)
        except dicom.errors.InvalidDicomError as e:
            print(
                f"Invalid DICOM file at index {idx}: {e}. Returning empty tensors.")
            return {'image': torch.zeros(1, 256, 256), 'mask': torch.zeros(1, 256, 256)}

        # Apply windowing
        patient_data = {
            'features': {
                'RescaleSlope': float(img_ds.RescaleSlope) if 'RescaleSlope' in img_ds else 1.0,
                'RescaleIntercept': float(img_ds.RescaleIntercept) if 'RescaleIntercept' in img_ds else 0.0
            },
            'img': img
        }
        windowed_img = window_image(
            patient_data=patient_data,
            organ='liver',
            w_width=self.hu_window.get('liver', (150, 30))[0],
            w_length=self.hu_window.get('liver', (150, 30))[1],
            rescale=True
        )
        image = windowed_img  # Now in [0, 255] as uint8
        mask = mask  # Assuming mask is binary or multi-class as needed

        # Normalize the image to [0, 1].
        image = image.astype(np.float32) / 255.0
        # Binary mask: 1 for tumor, 0 for background.
        mask = (mask > 0).astype(np.float32)

        # Add channel dimension.
        image = np.expand_dims(image, axis=0)  # Shape: (1, H, W)
        mask = np.expand_dims(mask, axis=0)    # Shape: (1, H, W)

        # Convert to torch tensors.
        sample = {'image': torch.tensor(image, dtype=torch.float32),
                  'mask': torch.tensor(mask, dtype=torch.float32)}

        if self.transform:
            sample = self.transform(sample)

        return sample


def load_data(extracted_dir, max_patients=None):
    """
    Loads and processes DICOM data into a list of patient data dictionaries.
    Implements lazy loading by storing file paths instead of loading all data into memory.

    Parameters:
    main_dir (str): Path to the main directory containing ZIP files.
    extracted_dir (str): Path where data will be extracted.
    max_patients (int, optional): Maximum number of patients to load. Useful for testing.

    Returns:
    list: List of patient data dictionaries.
    """

    patient_data_l = []

    patient_folders = [f for f in os.listdir(
        extracted_dir) if os.path.isdir(os.path.join(extracted_dir, f))]
    if max_patients:
        patient_folders = patient_folders[:max_patients]

    for patient_folder in patient_folders:
        try:
            _, patient_id = patient_folder.split('db1.')
            patient_id = int(patient_id)
        except ValueError:
            print(
                f"Skipping folder {patient_folder}: does not match expected naming convention.")
            continue

        img_dir = os.path.join(extracted_dir, patient_folder, 'PATIENT_DICOM')
        mask_dir_base = os.path.join(
            extracted_dir, patient_folder, 'MASKS_DICOM')

        if not os.path.exists(img_dir):
            print(
                f"Image directory {img_dir} does not exist. Skipping patient {patient_id}.")
            continue

        # Collect mask directories (e.g., 'livertumor', 'brain', etc.)
        mask_dirs = {}
        if os.path.exists(mask_dir_base):
            for mask_organ in os.listdir(mask_dir_base):
                mask_folder = os.path.join(mask_dir_base, mask_organ)
                if os.path.isdir(mask_folder):
                    mask_dirs[mask_organ] = mask_folder
        else:
            print(
                f"Mask directory {mask_dir_base} does not exist for patient {patient_id}.")
            # Optionally, decide whether to include patients without masks
            continue  # Skipping patients without masks

        # Append patient data
        patient_data = {
            'patient_id': patient_id,
            'img_dir': img_dir,
            'mask_dirs': mask_dirs  # List of mask directories
        }

        patient_data_l.append(patient_data)

    return patient_data_l
