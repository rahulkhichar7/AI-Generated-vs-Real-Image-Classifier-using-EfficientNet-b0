# 🧠 AI-Generated vs Real Image Classifier using EfficientNet-B0

This repository contains the final solution for the **HackRush Deep Fake Detection Challenge**, where the goal was to distinguish between AI-generated images and real human-generated images. Our final model, based on **EfficientNet-B0**, achieved an accuracy of **89.3%** on the final test set.

---

## 📌 Problem Statement

The task was to develop a robust and explainable deep learning model capable of classifying whether an image is AI-generated or real. The challenge required high accuracy and interpretability of the model's predictions.

---

## 🚀 Approach and Methodology

1. **Initial Experiments:**
   - Started with a custom CNN model (~40% accuracy).
   - Tried ResNet50 and ResNet-Freq models (~57% accuracy).
   - Performed comparative testing on a subset of 2000 images to evaluate each model's generalization.

2. **EfficientNet-B0 Implementation:**
   - Achieved ~70% accuracy out-of-the-box.
   - After hyperparameter tuning (optimizer, learning rate, batch size), reached 73%.
   - Trained on the full dataset leading to **91% training accuracy** and **89.3% final test accuracy**.

3. **Explainability:**
   - Integrated **Grad-CAM** to visualize class activation maps.
   - CAM visualizations demonstrate that the model attends to realistic textures and edges in real images and focuses on visual artifacts in AI-generated content.

---

## 📊 Results

| Metric               | Value         |
|----------------------|---------------|
| Training Accuracy    | 91%           |
| Final Test Accuracy  | 89.3%         |
| Model Used           | EfficientNet-B0 |
| Explainability Tool  | Grad-CAM      |

Kaggle Leaderboard : https://www.kaggle.com/competitions/hack-rush-deep-fake-detection/leaderboard
---

## 🗂️ Repository Structure

```
├── Final_Model_Code.ipynb        # Notebook containing full training & evaluation pipeline
├── HackRush Report.pdf           # Technical report with methodology, explainability, and analysis
├── final_submissions            # CSV files submitted for evaluation
├── GradCAM Visualizations/       # Heatmaps showing model attention regions
├── models/                       # Saved model weights
```

---

## 📷 Explainability Example

![GradCAM Examples]- See in report

> Grad-CAM helps confirm that the model learns interpretable patterns, focusing on textures and semantic objects in real images while detecting irregularities in AI-generated ones.

---

## 📌 Future Improvements

- Explore advanced frequency domain techniques.
- Try Vision Transformers (ViT) or hybrid CNN-Transformer models.
- Further improve robustness using adversarial training and ensemble models.

---

## 👤 Author

**Rahul Khichar**  
*B.Tech in AI, IIT Gandhinagar*  
GitHub: [@rahulkhichar7](https://github.com/rahulkhichar7)
