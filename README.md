# Attendance Marking Using Face Detection

## 📌 Project Overview

This project is a **Facial Recognition-based Attendance System** that automates the attendance marking process using OpenCV and Machine Learning. It detects faces in real-time from a webcam feed, recognizes registered users, and logs their attendance in a CSV file.

## 🎯 Features

- **Real-time Face Detection** using OpenCV's Haar Cascade Classifier.
- **Face Recognition** using K-Nearest Neighbors (KNN) classifier.
- **Automated Attendance Logging** in CSV files.
- **Flask Web Interface** for managing attendance.
- **User Registration** with face image collection.
- **Scalable and Secure** with proper data storage and access controls.

## 🖥️ Technologies Used

- **Python** (Main programming language)
- **OpenCV** (Face detection and image processing)
- **Flask** (Web framework for UI)
- **NumPy & Pandas** (Data handling and processing)
- **Scikit-learn** (Machine learning model - KNN)
- **Joblib** (Model serialization for efficient face recognition)

## 📂 Project Structure

```
Attendance-Marking-System/
│── Attendance/                      # Stores attendance CSV files
│── static/                           # Stores static assets (faces, models, etc.)
│   │── faces/                        # Folder for storing user face images
│   │── processed_background.png       # Processed background image
│   │── face_recognition_model.pkl     # Trained KNN model
│── templates/                         # HTML templates for Flask
│   │── home.html                      # Homepage template
│   │── add_attendance.html            # Template for adding new faces
│   │── mark_attendance.html           # Template for marking attendance
│── haarcascade_frontalface_default.xml # Haar Cascade for face detection
│── background.png                     # Background image
│── app.py                              # Main Flask application
│── requirements.txt                    # Dependencies for the project
│── README.md                           # Project documentation
│── .gitignore                          # Ignore unnecessary files
```

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/Kartik-Aswar/Attendance-Marking-Using-Face-Detection-.git
cd Attendance-Marking-Using-Face-Detection-
```

### 2️⃣ Create a Virtual Environment

```bash
python -m venv venv
```

Activate it:

- **Windows:** `venv\Scripts\activate`
- **Linux/Mac:** `source venv/bin/activate`

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Run the Application

```bash
python app.py
```

Visit [**http://127.0.0.1:5000/**](http://127.0.0.1:5000/) in your browser.

## 🖼️ Screenshots

### Home Page 
![image](https://github.com/user-attachments/assets/4dd3f1ba-3ab8-46b1-8137-16cb4b0d087e)




### Add New User
![image](https://github.com/user-attachments/assets/7a39b6db-73a0-4ab4-870c-91eafc3b219d)


### Mark Attendance
![image](https://github.com/user-attachments/assets/a1e88b31-c26a-4eb1-9f1e-1805d7225516)



## 🚀 Future Enhancements

- Improve recognition accuracy using **Deep Learning (CNNs, FaceNet, or ArcFace)**.
- Implement **multi-factor authentication** (Face + ID card or fingerprint).
- **Cloud-based Deployment** for scalability.
- Better handling of faces with masks, glasses, or low lighting.

## 📝 Authors

- **Kartik Aswar**
- **Hrushikesh Kale**
- **Kisan Yadav**

## 📜 License

This project is licensed under the **MIT License**.

---

Feel free to contribute or suggest improvements! 🚀

