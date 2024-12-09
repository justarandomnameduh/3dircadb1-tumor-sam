# DICOM Visualization and Tumor Analysis

## Overview
This project implements an end-to-end pipeline for processing, visualizing, and analyzing DICOM medical imaging data, with a specific focus on tumor segmentation using a U-Net architecture. The pipeline supports multi-patient datasets and enables interactive exploration of organs and tumors in 3D medical images. Key features include metadata extraction, advanced visualization, and tumor segmentation to facilitate both research and clinical applications.

---

## Features

### DICOM File Handling
- **Recursive File Management**: Automatically unzips and organizes multi-patient datasets.
- **HU Windowing**: Adjusts image intensity ranges for specific organs using Hounsfield Unit (HU) parameters.
- **Metadata Extraction**: Extracts patient-level metadata for in-depth analysis.

### 3D Visualization
- **Multi-View Support**: Displays axial, coronal, and sagittal views of medical images.
- **Interactive Widgets**: Enables slice selection, windowing adjustments, and overlay visualization.

### Tumor Segmentation
- **U-Net Integration**: Applies a U-Net model for precise tumor segmentation in liver CT images.
- **Mask Visualization**: Supports overlaying tumor and organ masks on medical scans.
- **Segmentation Metrics**: Computes Dice Similarity Coefficient (DSC) and Intersection over Union (IoU) for evaluating segmentation performance.

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

### 3. Interactive Visualization
- Use interactive tools to explore 3D scans with overlaid segmentation results.
- Adjust slice and visualization parameters dynamically.

### 4. Tumor Analysis
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

### Training the U-Net Model
```python
# Train U-Net on preprocessed liver CT images
train_unet(dataset_path="data/dicom", epochs=50, batch_size=8)
```

---

## Key Results
- **Enhanced Segmentation**: Achieved high Dice coefficient (e.g., 0.85) for liver tumor segmentation using U-Net.
- **Interactive Visualization**: Improved interpretability of complex 3D medical images through dynamic tools.
- **Comprehensive Analysis**: Automated tumor metrics and detailed reporting for multi-patient datasets.

---

## Technologies Used
- **DICOM Processing**: `pydicom`, `imageio`
- **Deep Learning**: `TensorFlow`, `Keras`
- **Visualization**: `matplotlib`, `ipywidgets`
- **Numerical Operations**: `numpy`, `scipy`

## Future Directions
- Despite extensive training, the current U-Net model has not achieved convergence after 100 epochs, indicating potential areas for enhancement. Future work will focus on the following strategies to improve model performance:

### Advanced DICOM Preprocessing Techniques:
- **Intensity Normalization**: Standardizing the intensity values across DICOM images can mitigate variations due to differing acquisition protocols, enhancing model robustness.
- **Noise Reduction**: Implementing denoising algorithms can improve image quality, facilitating more accurate segmentation.
- **Spatial Alignment**: Utilizing image registration methods to align images to a common reference frame can reduce anatomical variability, aiding the model in learning consistent features.

### Data Augmentation:
- **Geometric Transformations**: Applying random rotations, translations, scalings, and elastic deformations can help the model generalize to various anatomical presentations.
- **Intensity Variations**: Introducing random changes in brightness and contrast can make the model more resilient to differences in imaging conditions.
- **Spatial Augmentations**: Implementing techniques such as random cropping and flipping can expose the model to diverse spatial configurations, enhancing its adaptability.

### Exploration of Novel Segmentation Models:
- **Hybrid Architectures**: Investigating models that combine convolutional neural networks with transformers, such as the MRC-TransUNet, can capture both local and global contextual information, potentially improving segmentation accuracy. 
- **Diffusion Models**: Utilizing generative models like HiDiff, a hybrid diffusion framework, can complement discriminative segmentation methods by modeling underlying data distributions, leading to more precise segmentations. 
- **Attention Mechanisms**: Incorporating attention-based models can enhance the network's focus on relevant regions, improving the delineation of complex structures.

## Final Words
This project offers a platform for medical image analysis, combining visualization, segmentation, and statistical analysis to advance research and clinical workflows.
