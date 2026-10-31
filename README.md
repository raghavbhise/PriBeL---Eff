# PriBeL: Betel Leaf Disease Classification Using EfficientNetB0 and Grad-CAM

## Overview

This repository contains five reproducible experimental notebooks for classifying betel leaf images into three categories: **Diseased**, **Dried**, and **Healthy**. The experiments investigate the performance of an ImageNet-pretrained EfficientNetB0 model under two image-acquisition conditions:

- **Controlled Environment (CE):** Images captured in a controlled setting.
- **On-Field (OF):** Images captured under natural field conditions.

Each notebook runs the complete workflow independently for both environments. Five fixed random seeds are used to examine how model performance changes across different reproducible train/validation/test splits.

The work builds on the PriBeL betel-leaf dataset and the earlier PriBeL-Net study. It focuses on improving the EfficientNetB0 training pipeline and adding visual interpretability through Grad-CAM.

## Research Objectives

1. Develop an EfficientNetB0-based three-class betel leaf image classifier.
2. Evaluate the model independently on Controlled Environment and On-Field images.
3. Apply data augmentation and dropout regularization to support generalization.
4. Use adaptive learning-rate reduction, model checkpointing, and early stopping during training.
5. Repeat the experiments with five fixed random seeds to assess performance variability.
6. Evaluate classification performance using standard metrics and confusion matrices.
7. Use Grad-CAM heatmaps to visualize image regions that contribute to model predictions.

## Dataset

The experiments use the PriBeL betel-leaf image dataset, described in the accompanying research draft as containing **1,800 high-resolution JPEG images**:

| Class | Number of images |
|---|---:|
| Healthy | 669 |
| Diseased | 509 |
| Dried | 622 |
| **Total** | **1,800** |

The dataset contains images from both Controlled Environment and On-Field settings. The notebooks expect the dataset to be available in the Kaggle input directory.

### Dataset organization

The notebooks identify the two environment directories separately and construct image data frames from their class folders. The expected logical structure is:

```text
dataset/
├── Controlled Environment/
│   ├── Diseased/
│   ├── Dried/
│   └── Healthy/
└── On-Field/
    ├── Diseased/
    ├── Dried/
    └── Healthy/
```

Folder names and the Kaggle dataset path may need to be adjusted to match the dataset version attached to your Kaggle notebook.

### Data splitting

For each environment, the notebooks use a **stratified 80:10:10 train/validation/test split**:

- 80% training
- 10% validation
- 10% testing

The split is generated using `train_test_split` with the active random seed and class labels as the stratification target. Consequently, the exact samples in each split can differ between notebook runs, while class proportions are maintained as closely as possible.

## Model Architecture

The classifier uses **EfficientNetB0**, initialized with ImageNet-pretrained weights. The original ImageNet classification head is removed, and a task-specific classification head is added.

| Component | Configuration |
|---|---|
| Input | 224 × 224 × 3 RGB image |
| Backbone | EfficientNetB0 (`include_top=False`) |
| Pretrained weights | ImageNet |
| Backbone fine-tuning | Entire backbone trainable |
| Pooling | Global Average Pooling 2D |
| Dense layer | 128 units, ReLU |
| Regularization | Dropout (0.5) |
| Output layer | 3 units, Softmax |

The output layer predicts probabilities for the three target classes.

## Data Augmentation and Preprocessing

The notebooks include an augmentation pipeline designed to introduce variation in training images. The transformations include:

- Random horizontal flipping
- Random vertical flipping
- Random rotation (factor 0.2)
- Random zoom (factor 0.2)
- Random contrast adjustment (factor 0.2)

Images are processed at 224 × 224 pixels and use EfficientNet preprocessing/input scaling through the Keras image pipeline.

Augmentation is applied as part of the model's training pipeline; validation and test data are used for evaluation without training-time augmentation.

## Training Configuration

| Parameter | Configuration |
|---|---|
| Model | EfficientNetB0 |
| Optimizer | Adam |
| Initial learning rate | 0.0001 |
| Loss function | Categorical Cross-Entropy |
| Batch size | 32 |
| Maximum epochs | 25 |
| Early stopping patience | 7 epochs |
| Learning-rate reduction factor | 0.2 |
| Learning-rate reduction patience | 3 epochs |
| Minimum learning rate | 0.0000001 |
| Checkpoint monitoring | Validation loss |
| Checkpoint selection | Lowest validation loss |

Three callbacks are used:

1. **ModelCheckpoint:** Saves the model checkpoint when validation loss improves.
2. **EarlyStopping:** Stops training after seven epochs without validation-loss improvement and restores the best weights.
3. **ReduceLROnPlateau:** Reduces the learning rate when validation loss does not improve for three consecutive epochs.

A fresh EfficientNetB0 model is constructed for the On-Field experiment after the Controlled Environment evaluation. The notebooks also clear the Keras session and invoke garbage collection before the On-Field training stage to release resources and avoid reusing the Controlled Environment model.

## Experimental Notebooks

Five notebooks are included. Each notebook runs both environment-specific experiments.

| Notebook | Random seed | Controlled Environment | On-Field |
|---|---:|---|---|
| `pribel-efficientnetb0-seed-7-grad-cam.ipynb` | 7 | Train, evaluate and Grad-CAM | Train, evaluate and Grad-CAM |
| `pribel-efficientnetb0-seed-42-grad-cam.ipynb` | 42 | Train, evaluate and Grad-CAM | Train, evaluate and Grad-CAM |
| `pribel-efficientnetb0-seed-100-grad-cam.ipynb` | 100 | Train, evaluate and Grad-CAM | Train, evaluate and Grad-CAM |
| `pribel-efficientnetb0-seed-123-grad-cam.ipynb` | 123 | Train, evaluate and Grad-CAM | Train, evaluate and Grad-CAM |
| `pribel-efficientnetb0-seed-2024-grad-cam.ipynb` | 2024 | Train, evaluate and Grad-CAM | Train, evaluate and Grad-CAM |

The seed is the intended experimental variable across the five notebooks. Each notebook configures Python, NumPy, and TensorFlow random seeds using its corresponding value.

## Evaluation

Each environment-specific model is evaluated against its corresponding held-out test split. The notebooks report:

- Test loss
- Test accuracy
- Per-class precision
- Per-class recall
- Per-class F1-score
- Macro and weighted averages in the classification report
- Confusion matrix

Training and validation accuracy/loss curves are also plotted to inspect convergence and possible overfitting.

### Recorded test accuracy

The following values are taken from the executed notebook outputs. They are individual split results, not averages across seeds.

| Seed | Controlled Environment accuracy | On-Field accuracy |
|---:|---:|---:|
| 7 | 96.67% | 96.70% |
| 42 | 98.89% | 97.80% |
| 100 | 94.44% | 94.51% |
| 123 | 96.67% | 93.41% |
| 2024 | 100.00% | 97.80% |

These results show the observed test accuracy for each seed-specific split. They should not be interpreted as a single fixed performance guarantee. Because the random seed changes the split, each run evaluates on a potentially different set of test images.

## Model Interpretability with Grad-CAM

The notebooks implement **Gradient-weighted Class Activation Mapping (Grad-CAM)** to visualize the image regions that contribute to a model's predicted class.

The Grad-CAM workflow:

1. Extracts feature maps from the selected final convolutional layer (`top_conv`).
2. Computes gradients of the predicted class score with respect to those feature maps.
3. Uses pooled gradients as channel-importance weights.
4. Produces and normalizes a heatmap.
5. Resizes the heatmap to the input image dimensions and overlays it on the original image.

The notebooks display representative test images and their attention overlays for the Diseased, Dried, and Healthy classes. These visualizations provide an additional qualitative inspection of model behavior; they do not independently prove that the model has learned clinically or agronomically valid features.

## Technology Stack

- Python
- TensorFlow / Keras
- EfficientNetB0
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- OpenCV / image-processing utilities used in the notebooks
- Kaggle Notebooks
- NVIDIA Tesla T4 GPU environment (as described in the research draft)

## Running the Notebooks

1. Open Kaggle and create a new notebook.
2. Attach the PriBeL betel-leaf dataset to the notebook.
3. Confirm that the dataset directory paths in the data-ingestion cell match the attached dataset.
4. Select an accelerator with a compatible GPU, where available.
5. Open one of the five notebooks.
6. Run the cells sequentially, from environment setup through evaluation and Grad-CAM visualization.
7. Repeat with the remaining notebooks to execute all five seed-specific experiments.

The notebooks use Kaggle paths such as `/kaggle/input/` for dataset access and `/kaggle/working/` for generated model files. Runtime availability and GPU allocation depend on the Kaggle session.

## Reproducibility Notes

- The five configured seeds are **7, 42, 100, 123, and 2024**.
- Python hash, NumPy, and TensorFlow seeds are configured in each notebook.
- Stratified splitting is used to preserve class proportions.
- The architecture and principal training configuration are kept consistent across seed-specific runs.
- Each environment is trained independently.
- Training is bounded by 25 epochs, but early stopping may end a run sooner.

Fixed seeds improve experimental repeatability, but they do not guarantee bit-for-bit determinism across different hardware, TensorFlow/CUDA versions, or nondeterministic GPU operations. The five runs are a multi-seed stability assessment, **not k-fold cross-validation**.

## Limitations

The accompanying research draft identifies several limitations:

- The dataset images originate from a single geographic location.
- Image capture is limited to clear daylight conditions.
- The dataset consists of high-resolution images and does not fully represent camera noise, motion blur, or poor focus.
- The evaluation uses fixed seed-specific 80:10:10 splits rather than k-fold cross-validation.
- The current classification setup assigns images to one of three classes and does not address multiple co-occurring pathologies on a single leaf.

The results should therefore be interpreted within the scope of the dataset and experimental conditions.

## Future Scope

Potential directions identified in the research draft include:

- Evaluating generalization using images from multiple geographic locations, microclimates, and lighting conditions.
- Testing robustness to lower-quality images and varied backgrounds.
- Exploring multi-crop foliar datasets and broader plant-disease classification.
- Extending the three-class classifier toward a more comprehensive betel-leaf disease prediction system.
- Investigating a recommendation component for localized crop-care and treatment guidance.

## Repository Structure

```text
.
├── pribel-efficientnetb0-seed-7-grad-cam.ipynb
├── pribel-efficientnetb0-seed-42-grad-cam.ipynb
├── pribel-efficientnetb0-seed-100-grad-cam.ipynb
├── pribel-efficientnetb0-seed-123-grad-cam.ipynb
├── pribel-efficientnetb0-seed-2024-grad-cam.ipynb
└── README.md
```

Additional model checkpoints, plots, and generated outputs may be saved from the notebooks during execution; they are not assumed to be included in this repository structure.

## Acknowledgment

This work uses the PriBeL betel-leaf image dataset and is framed as a continuation of the research direction discussed in the PriBeL-Net study. Please consult the dataset's original source and publication for the applicable citation and usage terms.

---

**Note:** This README describes the five supplied notebooks and the accompanying research draft. Reported test accuracies are transcribed from the notebook execution outputs; they are not independently rerun as part of this documentation.
