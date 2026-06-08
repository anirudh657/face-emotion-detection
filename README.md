This project is a real-time facial emotion detection system built using Deep Learning (CNN) and Computer Vision.
It detects human emotions from images or webcam input and classifies them into different emotional states.

😃 Emotions Detected
Happy 😀
Sad 😢
Angry 😠
Neutral 😐
Surprise 😲
Fear 😨
Disgust 🤢
🚀 Features
Real-time emotion detection using webcam
Image-based emotion classification
Deep Learning model using CNN (Convolutional Neural Network)
Face detection using OpenCV
Label encoding for emotion classification
Trained model saved for reuse (.keras)
Simple Python-based application (emotion_app.py)
🛠️ Tech Stack
Python 🐍
TensorFlow / Keras 🤖
OpenCV 👁️
NumPy
Pandas
Scikit-learn
Jupyter Notebook 📓
📂 Project Structure
face-emotion-detection/
│
├── emotion_app.py              # Main application (webcam detection)
├── Emotion_Detection.ipynb     # Model training notebook
├── emotion_model.keras         # Trained CNN model
├── emotion_model_best.keras    # Best saved model
├── label_encoder.pkl           # Label encoder for classes
├── requirements.txt            # Dependencies
├── run_project.bat             # Run script
├── .gitignore                  # Ignored files
⚙️ Installation & Setup
1. Clone the repository
git clone https://github.com/anirudh657/face-emotion-detection.git
cd face-emotion-detection
2. Create virtual environment (optional)
python -m venv venv
venv\Scripts\activate   # Windows
3. Install dependencies
pip install -r requirements.txt
4. Run the project
python emotion_app.py
🎯 How It Works
Captures image from webcam
Detects face using OpenCV
Preprocesses the image
Passes it into CNN model
Predicts emotion class
Displays result on screen
📊 Model Architecture
Convolutional Neural Network (CNN)
Conv2D → ReLU → MaxPooling layers
Dropout for regularization
Dense layers for classification
Softmax output layer
📈 Future Improvements
Improve accuracy for similar emotions (Happy vs Neutral)
Add MediaPipe face landmarks
Deploy using Streamlit or Flask
Convert into mobile app
Optimize model for low-end devices
⚠️ Known Issues
Slight confusion between Neutral and Happy
Performance depends on lighting conditions
Model size is large (~65MB)
👨‍💻 Author

Anirudh
BTech CSE Student

GitHub: https://github.com/anirudh657

📜 License

This project is open-source and available under the MIT License.
