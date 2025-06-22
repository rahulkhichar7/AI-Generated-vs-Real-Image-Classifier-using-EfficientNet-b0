## 🧠 AI-Generated vs Human-Generated Image Classifier
Kaggle Compitition Link: https://www.kaggle.com/competitions/hack-rush-deep-fake-detection/leaderboard

This repository contains the complete solution for the HackRush Deep Fake Detection Challenge, where the objective was to build a robust, explainable model to distinguish AI-generated images from real, human-generated ones. The final model, based on EfficientNet-B0, achieved a test accuracy of **89.3%** and is accompanied by a detailed technical report and supporting files.

---

**Table of Contents**
- [Problem Statement](#problem-statement)
- [Approach & Methodology](#approach--methodology)
- [Model Architecture & Training](#model-architecture--training)
- [Explainability](#explainability)
- [Results](#results)
- [Repository Structure](#repository-structure)
- [Limitations & Future Work](#limitations--future-work)
- [Author](#author)

---

## Problem Statement

The challenge was to develop a deep learning pipeline that can accurately classify whether an image is AI-generated or real. The solution needed to be both high-performing and interpretable, providing insights into the model’s decision-making process[1][2].

---

## Approach & Methodology

- **Baseline Models:**  
  - Started with a custom CNN (4 conv layers), which set an initial benchmark (~40% accuracy).
  - Explored ResNet50 and a frequency-adapted ResNet ("ResNetFreq") for improved feature extraction, reaching ~57% accuracy.
  - Used a subset of 2,000 images for rapid prototyping and model comparison[1][2].

- **EfficientNet-B0 Adoption:**  
  - Chosen for its compound scaling and efficiency.
  - Outperformed previous models, achieving 70% accuracy out-of-the-box; after hyperparameter tuning and augmentations, reached 73% on validation.
  - Final training on the complete dataset led to 91% internal validation accuracy and 89.3% on the official test set[1][2].

- **Training Details:**  
  - Optimizer: Adam
  - Loss: CrossEntropyLoss
  - Learning Rate: 1e-4 (cosine annealing)
  - Batch Size: 128
  - Epochs: 10
  - Input Size: 224×224
  - Augmentations: Random flips, rotations, color jitter[1].

---

## Model Architecture & Training

- **Custom CNN:** Served as a baseline; underperformed due to limited capacity.
- **ResNet50 & ResNetFreq:** Leveraged transfer learning and frequency-domain preprocessing; moderate gains.
- **EfficientNet-B0:** Provided the best trade-off between performance and computational efficiency, making it the final choice[1].

---

## Explainability

- **Grad-CAM Visualizations:**  
  - Implemented Grad-CAM to highlight which regions of each image influenced the model’s predictions.
  - Observed that the model attends to textures and artifacts in AI-generated images, and to natural boundaries and color distributions in real images.
  - These visualizations confirm that the model is learning meaningful, interpretable features and not overfitting to noise[1][2].

---

## Results

| Model           | Subset Accuracy | Final Validation | Test Accuracy |
|-----------------|-----------------|------------------|--------------|
| Custom CNN      | ~40%            | —                | —            |
| ResNet50        | ~57%            | —                | —            |
| ResNetFreq      | ~58%            | —                | —            |
| EfficientNet-B0 | 73%             | 91%              | 89.3%        |

EfficientNet-B0 outperformed all other models, demonstrating strong generalization and robustness[1].

---

## Repository Structure

```
├── Best_model_selection.ipynb                        # Model selection and comparison notebook
├── Efficientnet_Training.ipynb                       # EfficientNet-B0 training pipeline
├── best_model_4_epoch_8_train_98.37                  # Model checkpoint (receved best train accuracy of 98% in epoch 8 with test accracy of 91%)
                                                      # Data was devided in four parts, "best_model_4_epoch_8_train_98.37" here 4 denotes that model is being trained on full dataset (60K images)
├── Final submission csv file.csv                     # Final predictions for submission
├── Final_model_traning_&prediction_&explanibility.ipynb  # Final training and inference notebook
├── HackRush Report.pdf                                   # Technical report with methodology and analysis
├── README.md                                             # Project documentation (this file)
```
[As per the attached image and files][3].

---

## Limitations & Future Work

**Known Limitations:**
- Current model is tailored to the provided dataset; may not generalize to newer AI-generation techniques or different image domains.
- High-quality AI images with minimal artifacts can still confuse the classifier.
- Potential dataset imbalance and image quality issues may affect performance[1].

**Potential Improvements:**
- Integrate frequency-domain features (e.g., DCT coefficients) in a multimodal setup.
- Experiment with ensemble models combining EfficientNet and ResNetFreq.
- Explore adversarial training for better robustness.
- Fine-tune larger EfficientNet variants (B3/B4) and introduce more synthetic augmentations.
- Investigate Vision Transformers (ViT) or hybrid CNN-Transformer architectures[1][2].

---

## Author

**Rahul Khichar**  
B.Tech in AI, IIT Gandhinagar  
GitHub: [@rahulkhichar7](https://github.com/rahulkhichar7)[2]

---

For detailed methodology, results, and visualizations, refer to the [HackRush Report.pdf][1].

---

> “Grad-CAM visualizations confirm that the model learns interpretable and relevant features, focusing on meaningful regions to distinguish between AI-generated and real images.”
