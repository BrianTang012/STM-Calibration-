# STM Tip Calibration

This repository provides code for two STM tip calibration methods, both optimized for Au(111) surfaces. Each method has specific requirements and compatibility notes detailed below.

## Methods Overview

### Method 1: DeepSPM Classifier-Based Approach
- **Description:** Code for the DeepSPM classifier model used in STM tip calibration.
- **Requirements:** 
  - Compatible with Python 3.6 to 3.9
  - Requires TensorFlow 1.x
- **Note:** This method may require setup in an older Python environment due to its reliance on TensorFlow 1.x.

### Method 2: Custom ML Method Inspired by DeepSPM
- **Description:** A custom machine learning approach inspired by DeepSPM, designed for STM tip calibration. This method uses controlled crashes as the primary conditioning action to recalibrate the tip and improve image quality during scanning, with a focus on simplicity and reproducibility.
- **Requirements:**
  - Recommended Python version: 3.11 or higher
  - Requires only PyTorch

## Important Note
Both Method 1 and Method 2 are based on earlier code versions. They may require updates to integrate the latest enhancements for optimal performance.
