# 🎓 Smart Attendance System Using Face Recognition

### Face Embeddings • Webcam • CSV Attendance Logging

## 📌 Overview

The Smart Attendance System is a computer vision–based application that automatically marks attendance using real-time face recognition via a webcam.

The system uses face embeddings generated from pre-collected face images stored in labeled folders (e.g., data/Trisha, data/Aishwarya).
When a person appears in front of the webcam, their face is recognized and their attendance is recorded in a CSV file with timestamp.

This eliminates manual attendance, proxy attendance, and human error.

## 🎯 Key Features

📷 Real-time face detection using webcam

🧠 Face recognition using face embeddings

📁 Folder-based dataset structure (person-wise)

📝 Automatic attendance marking in CSV file

⏰ Timestamped attendance records

🚫 Prevents duplicate attendance entries

## 🧠 Core Concepts Used

Computer Vision

Face Detection

Face Embeddings

Similarity Matching (Cosine / Euclidean)

Real-Time Video Processing

Attendance Automation

## 🗂️ Dataset Structure

Images are organized in person-specific folders:

data/
├── Trisha/
│   ├── img1.jpg
│   ├── img2.jpg
│
├── Aishwarya/
│   ├── img1.jpg
│   ├── img2.jpg


Each folder name represents the person’s identity.

## 🛠️ Tech Stack

Python

OpenCV

face_recognition / dlib / MediaPipe

NumPy

Pandas

CSV File Handling

VS Code

## ⚙️ Working Pipeline

Load face images from dataset folders

Generate face embeddings for each person

Store known face encodings with names

Activate webcam

Detect face in real-time

Compare detected face embedding with known embeddings

If matched:

Display name on screen

Mark attendance in CSV (once per session)

## 📊 Attendance Output Format (CSV)
Name,Date,Time
Trisha,2025-12-25,09:30:15
Aishwarya_Rai,2025-12-25,09:31:02

## 🚀 Applications

🏫 Colleges & Schools

🏢 Offices & Organizations

🧪 Labs & Training Centers

🏆 Events & Workshops

🔮 Future Enhancements

📌 Database integration (MySQL / MongoDB)

🌐 Web dashboard (Flask / Streamlit)

😷 Masked face recognition

📱 Mobile camera support

🔐 Role-based access control

## 🧑‍💻 Author

Trisha
Engineering Student | AI & Computer Vision Enthusiast
Passionate about building intelligent real-world systems.
