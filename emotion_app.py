import cv2
import numpy as np
from tensorflow.keras.models import load_model
from collections import deque, Counter

model = load_model("emotion_model_best.keras")

labels = [
    'angry',
    'disgust',
    'fear',
    'happy',
    'neutral',
    'sad',
    'surprise'
]

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)
cap = cv2.VideoCapture(0)

cap.set(3, 640)
cap.set(4, 480)
emotion_buffer = deque(maxlen=10)

last_label = "neutral"

while True:

    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    gray = cv2.equalizeHist(gray)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=6,
        minSize=(80,80)
    )

    if len(faces) > 0:

        faces = sorted(
            faces,
            key=lambda x: x[2] * x[3],
            reverse=True
        )

        faces = [faces[0]]

    for (x, y, w, h) in faces:

        face = gray[y:y+h, x:x+w]

        try:
            face = cv2.resize(
                face,
                (48,48)
            )
        except:
            continue

        face = cv2.equalizeHist(face)

        face = cv2.GaussianBlur(
            face,
            (3,3),
            0
        )

        face = face.astype(
            "float32"
        ) / 255.0

        face = face.reshape(
            1,
            48,
            48,
            1
        )

        pred = model.predict(
            face,
            verbose=0
        )[0]

        idx = np.argmax(pred)

        confidence = float(
            pred[idx]
        )

        label = labels[idx]

        if confidence < 0.60:
            label = last_label
        else:
            last_label = label

        emotion_buffer.append(
            label
        )

        final_label = Counter(
            emotion_buffer
        ).most_common(1)[0][0]

        text = (
            f"{final_label} "
            f"({confidence*100:.1f}%)"
        )

        cv2.rectangle(
            frame,
            (x,y),
            (x+w,y+h),
            (0,255,0),
            2
        )

        cv2.putText(
            frame,
            text,
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255,255,255),
            2
        )

    cv2.imshow(
        "Emotion Detection",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()