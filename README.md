# 🐱🐶 Cats vs Dogs Image Classification

This project is a **deep learning model** that classifies images of cats and dogs using **Transfer Learning (MobileNetV2)**.  
It leverages pre-trained ImageNet features for improved accuracy, even on a small dataset.

---

## 📂 Dataset
- Two folders:
  - `cats/` → contains cat images
  - `dogs/` → contains dog images
- The dataset is split into **training (80%)** and **validation (20%)** automatically.

Example structure:
```
dataset/
│── cats/
│   ├── cat_1.jpg
│   ├── cat_2.jpg
│── dogs/
    ├── dog_1.jpg
    ├── dog_2.jpg
```

---

## ⚙️ Features
- Uses **MobileNetV2** (pre-trained on ImageNet).
- Data augmentation for robust training.
- Early stopping to prevent overfitting.
- Confusion matrix and classification report.
- Interactive prediction: test your own images after training.

---

## 🚀 How to Run
1. Clone the repo:
   ```bash
   git clone https://github.com/yourusername/cats-vs-dogs-classification.git
   cd cats-vs-dogs-classification
   ```

2. Install dependencies:
   ```bash
   pip install tensorflow matplotlib seaborn scikit-learn
   ```

3. Place your dataset inside a folder:
   ```
   dataset/cats/
   dataset/dogs/
   ```

4. Run the script:
   ```bash
   python Image_Classification.py
   ```

---

## 🖼️ Testing
After training, you can test new images:
```
Enter image path: dataset/cats/cat_20.jpg
🐱 cat_20.jpg -> Cat (0.94)

Enter image path: dataset/dogs/dog_15.jpg
🦮 dog_15.jpg -> Dog (0.89)
```

---

## 📊 Results
- Training Accuracy: ~85–95% (with transfer learning)
- Validation Accuracy: ~80–90%
- Confusion Matrix and classification report included.
<img width="1302" height="512" alt="image" src="https://github.com/user-attachments/assets/4bffa77c-c7c4-468c-871a-75f703905d82" />

---

## 📌 Future Improvements
- Fine-tune last MobileNetV2 layers for >95% accuracy.
- Deploy as a Flask/Django web app.
- Extend to more animal classes.

