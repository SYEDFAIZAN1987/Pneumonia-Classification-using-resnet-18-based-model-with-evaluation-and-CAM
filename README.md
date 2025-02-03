# 🦠 Pneumonia Classification Model

Welcome to the **Pneumonia Classification Model** repository! This project utilizes a deep learning approach based on the powerful **ResNet-18** architecture to classify chest X-ray images and detect pneumonia with high accuracy.

![Pneumonia Detection](https://github.com/SYEDFAIZAN1987/Pneumonia-Classification-using-resnet-18-based-model-with-evaluation-and-CAM/blob/main/CAM%20picture.png)

## 🚀 Project Overview

This repository contains code for:

- Training a **ResNet-18** model for pneumonia classification
- Evaluating model performance with detailed accuracy metrics
- Visualizing key features using **Class Activation Mapping (CAM)**

## 📊 Key Features

✅ **ResNet-18 Backbone**: Fine-tuned on chest X-ray images for accurate pneumonia detection  
✅ **Model Evaluation**: Includes both weighted and non-weighted accuracy assessments  
✅ **Class Activation Mapping (CAM)**: Visualizes critical regions influencing the model’s predictions  
✅ **Efficient Training**: Optimized data pipelines for fast model training and inference

---

## 🗂️ Repository Structure

```
Pneumonia-Classification-Model/
├── PneumoniaClassification.ipynb   # Jupyter Notebook for model training & evaluation
├── checkpoints/                    # Saved model weights
├── data/                           # Dataset folder (chest X-ray images)
├── outputs/                        # Generated CAM images & evaluation results
└── README.md                       # Project documentation
```

---

## ⚙️ Installation

1. **Clone the Repository**:
```bash
git clone https://github.com/SYEDFAIZAN1987/Pneumonia-Classification-using-resnet-18-based-model-with-evaluation-and-CAM.git
cd Pneumonia-Classification-Model
```

2. **Install Dependencies**:
```bash
pip install -r requirements.txt
```

3. **Run the Model**:
```bash
jupyter notebook PneumoniaClassification.ipynb
```

---

## 🔍 Model Performance

### **Non-Weighted Accuracy:**

![Non-Weighted Accuracy](https://github.com/SYEDFAIZAN1987/Pneumonia-Classification-using-resnet-18-based-model-with-evaluation-and-CAM/blob/main/Non%20Weighted%20Accuracy.png)

### **Weighted Accuracy:**

![Weighted Accuracy](https://github.com/SYEDFAIZAN1987/Pneumonia-Classification-using-resnet-18-based-model-with-evaluation-and-CAM/blob/main/Weighted%20Accuracy.png)

### **Class Activation Mapping (CAM):**

Visual representation of regions critical to the model’s pneumonia classification:

![CAM Visualization](https://github.com/SYEDFAIZAN1987/Pneumonia-Classification-using-resnet-18-based-model-with-evaluation-and-CAM/blob/main/CAM%20picture.png)

---

## 📈 Evaluation Metrics

- **Precision, Recall, F1-Score** for performance evaluation
- **Confusion Matrix** for classification analysis
- **Weighted vs Non-Weighted Accuracy** comparison

---

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repository  
2. Create a new branch: `git checkout -b feature-branch`  
3. Commit your changes  
4. Open a pull request 🚀

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

---

## 👨‍⚕️ Author

Developed by **Syed Faizan**  
For any queries or collaborations, feel free to connect on [GitHub](https://github.com/SYEDFAIZAN1987).

⭐ **If you found this project useful, please give it a star!** ⭐
