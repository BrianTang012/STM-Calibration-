# STM Tip Calibration

This repository provides code for the **Method 2: Custom ML Method** designed for STM tip calibration, optimized for Au(111) surfaces. Method 2 is inspired by DeepSPM but focuses on simplicity and reproducibility. The scripts for Method 2 are shared in this repository. However, **Method 1 is based on the open-source DeepSPM project** (available at [DeepSPM GitHub](https://github.com/abred/DeepSPM)), with modifications and optimizations specific to our STM setup and project needs.

This work is part of my **NTU Final Year Project** for graduation, where I aim to integrate machine learning for autonomous STM tip calibration.

## Methods Overview

### Method 1: DeepSPM Classifier-Based Approach (Modified)
- **Description:** Code for the DeepSPM classifier model used in STM tip calibration. This method has been calibrated and optimized for our STM and project requirements.
- **Requirements:**
  - Compatible with Python 3.6 to 3.9
  - Requires TensorFlow 1.x
- **Note:** This method is a modified version of the open-source DeepSPM repository and may require setup in an older Python environment due to its reliance on TensorFlow 1.x.

### Method 2: Custom ML Method Inspired by DeepSPM
- **Description:** A custom machine learning approach inspired by DeepSPM, designed for STM tip calibration. This method uses controlled crashes as the primary conditioning action to recalibrate the tip and improve image quality during scanning, with a focus on simplicity and reproducibility.
- **Requirements:**
  - Recommended Python version: 3.11 or higher
  - Requires only PyTorch

## Important Note
Both Method 1 and Method 2 are based on earlier code versions and may require updates to integrate the latest enhancements for optimal performance.

