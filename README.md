# DICOM Visualization and Tumor Analysis

## Overview
This project implements a comprehensive pipeline for visualizing and analyzing DICOM medical imaging data. It includes features for processing, visualizing, and extracting metadata from 3D medical images. Designed for multi-patient datasets, the project allows interactive exploration and analysis of organs and tumors within medical scans.

---

## Features

### DICOM File Handling
- **Recursive File Management**: Automatically unzips and organizes multi-patient datasets.
- **HU Windowing**: Adjusts image intensity ranges for specific organs using Hounsfield Unit (HU) parameters.
- **Metadata Extraction**: Extracts patient-level metadata for in-depth analysis.

### 3D Visualization
- **Multi-View Support**: Displays axial, coronal, and sagittal views of medical images.
- **Interactive Widgets**: Enables slice selection, windowing adjustments, and overlay visualization.

### Tumor Analysis
- **Mask Integration**: Visualizes organ-specific and tumor-specific masks.
- **Tumor Metrics**: Calculates tumor count, size, and scale relative to the liver for each patient.
- **Statistical Reporting**: Outputs tumor statistics for detailed patient analysis.

---

## Workflow

### 1. Data Preparation
- Extract and organize multi-patient DICOM datasets.
- Compute scaling factors and apply HU windowing for specific organs.

### 2. Interactive Visualization
- Explore 3D images with overlays using interactive widgets.
- Adjust slices and windowing parameters dynamically.

### 3. Tumor Analysis
- Visualize organ and tumor masks.
- Compute and display tumor statistics, including size and count.

---

## Example Usage

### Visualizing DICOM Slices
```python
# Visualize liver slices for a specific patient
dicom_interact(patient_id=1, mask_organ="liver")
```

### Analyzing Tumor Data
```python
# Display tumor statistics for all patients
for i in range(1, num_patient):
    print(f"Patient 1.{i}: Tumor count = {metadata[i]['tumor_count']}, Tumor size = {metadata[i]['tumor_size']}")
```

---

## Key Results
- **Interactive Visualization**: Improved interpretability of complex 3D medical images.
- **Tumor Insights**: Automated computation of tumor metrics for multi-patient datasets.
- **Comprehensive Analysis**: Support for multi-organ visualization and analysis. 

---

## Technologies Used
- **DICOM Processing**: `pydicom`, `imageio`
- **Visualization**: `matplotlib`, `ipywidgets`
- **Numerical Operations**: `numpy`, `scipy`

This project is a robust tool to explore and analyze medical imaging datasets effectively.
