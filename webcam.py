from tensorflow.keras.models import load_model
import cv2
from tensorflow.keras.applications import MobileNetV2
from datetime import datetime
import numpy as np
import pickle
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import pandas as pd


with open("face_classifier.pkl", "rb") as f:
    classifier = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

with open("normalizer.pkl", "rb") as f:
    normalizer = pickle.load(f)

with open('class_names.pkl','rb') as f:
    class_names = pickle.load(f)

embedding_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3),
    pooling="avg"
)

attendance = {}
cap = cv2.VideoCapture(0)


face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face = cv2.resize(face, (224, 224))
        face = preprocess_input(face.astype("float32"))

        face = np.expand_dims(face, axis=0)

        # 🔥 EMBEDDING
        embedding = embedding_model.predict(face, verbose=0)
        embedding = normalizer.transform(embedding)

        # 🔥 CLASSIFICATION
        probs = classifier.predict_proba(embedding)[0]
        class_idx = np.argmax(probs)
        confidence = probs[class_idx]
        name = label_encoder.inverse_transform([class_idx])[0]

        if confidence > 0.7:
            cv2.putText(frame, f"{name} {confidence*100:.2f}%",
                        (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (0, 255, 0), 2)

            cv2.rectangle(frame, (x, y), (x+w, y+h),
                          (0, 255, 0), 2)

            if name not in attendance:
                attendance[name] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    cv2.imshow("attendance_system", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()


csv_file = "attendance.csv"
import os
# Load existing attendance if file exists
if os.path.exists(csv_file):
    df = pd.read_csv(csv_file)
else:
    df = pd.DataFrame(columns=["Name", "Time"])

# After webcam loop, convert session attendance to DataFrame
session_df = pd.DataFrame(list(attendance.items()), columns=["Name", "Time"])

# Merge, avoid duplicates
df = pd.concat([df, session_df])

# Save back to CSV
df.to_csv(csv_file, index=False)
print("✅ Attendance updated in single CSV")