# 😷 Robust Face Mask Detection System

A robust, multi-model machine learning application designed to detect face masks in real-time. This system is engineered to overcome common computer vision challenges—such as lighting bias and overfitting—by implementing advanced pre-processing (CLAHE) and dimensionality reduction (PCA).

The system trains three distinct classifiers: **Support Vector Machine (SVM)**, **Random Forest**, and **Logistic Regression**, allowing users to benchmark and switch between them dynamically.

---

## 🚀 Key Features

* **Smart State Management:** Automatically loads saved models (`.pkl`) on startup. You only train once; subsequent runs are instant.
* **Robust Pre-processing:** Uses **CLAHE** (Contrast Limited Adaptive Histogram Equalization) to normalize shadows and lighting. This prevents the model from mistaking a dark chin or beard for a mask.
* **Overfitting Protection:** Implements **PCA** (Principal Component Analysis) to retain only 95% of relevant variance, filtering out noise and background details that confuse the model.
* **Multi-Model Architecture:** Trains three models simultaneously. Users can switch the "active" model during runtime to compare performance.
* **Live Confidence Thresholding:** Real-time webcam testing with a user-adjustable probability threshold. If the model isn't sure, it reports "Uncertain" rather than guessing.
* **Interactive CLI Menu:** A fully featured command-line interface for testing, retraining, and configuration.

---

## 📂 Project Structure

The code expects the following directory structure:

```text
Mask Detection
├── main.py                      # The primary application script
├── mask_detector_all_models.pkl # The saved model state (generated after first run)
├── model_comparison.png         # Auto-generated accuracy chart
├── dataset1/                    # Data directory
│   ├── Train/
│   │   ├── WithMask/
│   │   └── WithoutMask/
│   ├── Validation/
│   │   ├── WithMask/
│   │   └── WithoutMask/
│   └── Test/
│       ├── WithMask/
│       └── WithoutMask/
```
---

🛠️ Technical Implementation Details

1. Pre-Processing Pipeline (extract_features)
This function transforms raw pixels into meaningful data.

* Grayscale Conversion: Reduces computational load by removing color information, focusing purely on structure/texture.
* CLAHE (Contrast Limited Adaptive Histogram Equalization):
    * Reason: Standard image equalization creates noise in flat areas. CLAHE enhances local contrast. This is critical for distinguishing a mask's edge from a shadow cast by the nose or jawline.
* HOG (Histogram of Oriented Gradients):
    * Reason: Extracts the shape and edge direction. Masks have distinct horizontal upper edges and smooth textures compared to a human mouth.

2. Dimensionality Reduction (train_system)
* StandardScaler: Normalizes feature values so that no single feature dominates the objective function (essential for SVM convergence).
* PCA (Principal Component Analysis):
    * Reason: HOG produces thousands of features. PCA reduces this to a smaller set that explains 95% of the variance. This prevents the "curse of dimensionality" and stops the model from memorizing specific training images (overfitting).

3. The Models
* SVM (RBF Kernel): Uses a non-linear kernel to separate complex data points. Configured with class_weight='balanced' to handle uneven datasets.
* Random Forest: An ensemble of decision trees. Depth is limited (max_depth=8) to force generalization rather than memorization.
* Logistic Regression: A linear classifier with strong regularization (C=0.01) acting as a conservative baseline.

---

🎮 User Guide (Interactive Menu)
When you run main.py, the following options are available:

Testing Options
1. Test Custom Image: Input a file path to classify a single image.
2. Test Directory: Batch process an entire folder of images and print results to the console.
3. Live Camera Test: Opens the webcam.
    * Visuals: Green box = With Mask, Red box = Without Mask, Yellow box = Uncertain.
    * Controls: Press q to close the camera window.
4. Show Confusion Matrix: Displays a heatmap showing the True Positives vs. False Negatives for the currently active model.

Configuration Options
5. Change Active Model: Instantly swap the prediction engine (e.g., from SVM to Random Forest) to see which works best for your specific face/lighting.
6. Change Probability Threshold: Adjust how confident the model must be to make a decision (Default: 0.75). Increase this if you see too many false positives.

System Options
7. RETRAIN MODELS: Forces the system to re-load all images from the dataset1 folder and re-train all models. Use this if you have added new data.
8. RELOAD SAVED MODELS: Re-reads the .pkl file from the disk, discarding any unsaved changes to the active model or threshold.
9. Exit: Closes the application.

---

📊 Outputs
Upon successful training, the system generates:
* model_comparison.png: A bar chart comparing the accuracy of the three models.
* confusion_matrix_[ModelName].png: Individual images showing the classification metrics for each model.
* mask_detector_all_models.pkl: A binary file containing the trained models, the scaler, and the PCA object.

📦 Dependencies
* Python 3.x
* OpenCV (cv2)
* Scikit-Learn (sklearn)
* Scikit-Image (skimage)
* Numpy
* Matplotlib & Seaborn (for visualization)

-----
