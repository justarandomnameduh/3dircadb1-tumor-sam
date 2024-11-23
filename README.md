# Tumor Segmentation with DICOM Processing and SAM Integration using 3Dircadb1 dataset

## Overview

This repository implements a system for **tumor segmentation** from **DICOM medical images**, utilizing **Segment Anything Model (SAM)** for precise mask generation. The project facilitates end-to-end processing of 3D medical imaging data, including DICOM file handling, visualization, and segmentation techniques. Interactive widgets and visualization functionalities enhance usability for medical professionals and researchers.

---

## Features

### Data Preparation
- **DICOM Processing**: Handles DICOM data, extracting slices and metadata for analysis.
- **Recursive File Handling**: Unzips and organizes complex directory structures for multi-patient datasets.
- **Windowing**: Applies Hounsfield Unit (HU) windowing to adjust image intensity ranges for various organs (e.g., liver, lungs).

### Interactive Visualization
- **3D Medical Imaging**: Displays axial, coronal, and sagittal views with optional mask overlays.
- **Widget Integration**: Enables user interaction for slice selection, windowing adjustments, and segmentation visualization.

### Tumor Segmentation
- **SAM Integration**: Leverages **Segment Anything Model (SAM)** to segment tumor regions in specific slices with RGB input processing.
- **Automatic Mask Generation**: Dynamically generates segmentation masks for tumor identification.
- **Organ Masking**: Utilizes organ-specific masks for enhanced localization.

### Metadata Extraction
- **Patient-Level Data**: Computes metrics such as tumor count, size, and scale relative to organ size.
- **Statistical Reporting**: Provides detailed summaries of tumor characteristics for each patient.

---

## Workflow

### 1. Data Ingestion and Processing
- Extract and organize DICOM files from a multi-patient dataset.
- Apply HU windowing to enhance image quality for specific organs.
- Compute scaling factors for accurate multi-dimensional visualization.

### 2. Interactive Visualization
- Visualize and interact with 3D images using widgets for slice navigation and intensity adjustments.
- Overlay organ-specific masks on image slices for detailed analysis.

### 3. Tumor Segmentation with SAM
- Perform automatic segmentation of tumor regions using SAM.
- Generate and display segmentation masks overlaid on DICOM slices.

---

## Example Usage

### 1. Extract and Process Data
```python
# Extract data and load DICOM files
MAIN_DIR = '/path/to/dataset'
```

### 2. Visualize DICOM Slices
```python
# Visualize liver slices for a specific patient
dicom_interact(1, "liver")
```

### 3. Segment Tumors with SAM
```python
# Segment tumors for a given patient and slice
dicom_interact_with_sam(1, "liver")
```

---

## Key Results

- **Tumor Metrics**: Extracted tumor count, size, and scale for each patient.
- **Interactive Visualization**: Widgets and overlays significantly improve the interpretability of medical images.

---

## Limitations

- SAM relies on automated mask generation without allowing manual selection of points or fine-tuning. This can lead to inaccuracies in organ boundaries, especially in cases with complex anatomical structures or low-contrast regions.
- SAM is a general-purpose segmentation model and is not fine-tuned for medical imaging or specific organs, which can result in suboptimal mask generation.

## References

- **SAM**: Segment Anything Model by Meta AI [GitHub](https://github.com/facebookresearch/segment-anything)
- **Pydicom**: DICOM processing library [Docs](https://pydicom.github.io/)
