 # 😊 Face Emotion Detection using Deep Learning

A real-time facial emotion detection system built using **TensorFlow**, **Keras**, and **OpenCV**. The application detects faces from webcam input and classifies human emotions using a Convolutional Neural Network (CNN).

---

## 📌 Overview

This project uses Computer Vision and Deep Learning techniques to recognize facial expressions and classify them into different emotional categories.

The model is trained on facial expression datasets and can perform emotion recognition in real time through a webcam feed.

---

## 🎯 Emotions Detected

- 😀 Happy
- 😢 Sad
- 😠 Angry
- 😐 Neutral
- 😲 Surprise
- 😨 Fear
- 🤢 Disgust

---

## 🚀 Features

- Real-time webcam emotion detection
- CNN-based emotion classification
- Face detection using OpenCV
- Trained deep learning model saved for reuse
- Label encoding for emotion prediction
- Easy-to-run Python application

---

## 🛠️ Tech Stack

- Python
- TensorFlow
- Keras
- OpenCV
- NumPy
- Pandas
- Scikit-learn
- Jupyter Notebook

---

## 📂 Project Structure

```text
face-emotion-detection/
│
├── emotion_app.py
├── Face_Emotion_Detection_Training.ipynb
├── emotion_model.keras
├── emotion_model_best.keras
├── label_encoder.pkl
├── requirements.txt
├── run_project.bat
├── README.md
└── .gitignore
```

---

## ⚙️ Installation

### Clone Repository

```bash
git clone https://github.com/anirudh657/face-emotion-detection.git
cd face-emotion-detection
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

```bash
python emotion_app.py
```

The webcam will open and display the detected emotion in real time.

---

## 🧠 Model Architecture

The emotion classification model is based on a Convolutional Neural Network (CNN) consisting of:

- Convolution Layers
- ReLU Activation
- Max Pooling Layers
- Dropout Layers
- Dense Layers
- Softmax Output Layer

---

## 🔄 Workflow

1. Capture image from webcam
2. Detect face using OpenCV
3. Preprocess facial image
4. Pass image to trained CNN model
5. Predict emotion
6. Display emotion label on screen

---

## 📸 Output Preview

Add screenshots here after taking output images.

Example:

```markdown
![Output](images/output.png)
```

---

## 📈 Future Improvements

- Improve accuracy for similar emotions
- Add MediaPipe face landmarks
- Deploy using Streamlit
- Create web-based interface
- Optimize model for faster inference

---

## ⚠️ Limitations

- Performance may vary under poor lighting conditions
- Similar expressions can occasionally be confused
- Accuracy depends on face visibility and camera quality

---

## 👨‍💻 Author

**Anirudh**

B.Tech CSE Student

GitHub:
https://github.com/anirudh657

---

## ⭐ Support

If you found this project useful, consider giving it a star on GitHub.
