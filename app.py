import cv2
import os
from flask import Flask, request, render_template, Response
from datetime import date
from datetime import datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib


app = Flask(__name__)

nimgs = 10

imgBackground=cv2.imread("background.png")
cv2.imwrite('static/processed_background.png', imgBackground)

datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")


face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')



# Load the face detection model (Haar Cascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Global variables
image_count = 0
captured_images = []
is_capturing = False  # Global flag to indicate capturing state



if not os.path.isdir('Attendance'):
    os.makedirs('Attendance')
if not os.path.isdir('static'):
    os.makedirs('static')
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')
if f'Attendance-{datetoday}.csv' not in os.listdir('Attendance'):
    with open(f'Attendance/Attendance-{datetoday}.csv', 'w') as f:
        f.write('Name,Roll,Time')

def totalreg():
    return len(os.listdir('static/faces'))

def extract_faces(img):
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_points = face_detector.detectMultiScale(gray, 1.2, 5, minSize=(20, 20))
        return face_points
    except:
        return []

def identify_face(facearray):
    model = joblib.load('static/face_recognition_model.pkl')
    return model.predict(facearray)


def train_model():
    faces = []
    labels = []
    userlist = os.listdir('static/faces')
    for user in userlist:
        for imgname in os.listdir(f'static/faces/{user}'):
            img = cv2.imread(f'static/faces/{user}/{imgname}')
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face.ravel())
            labels.append(user)
    faces = np.array(faces)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces, labels)
    joblib.dump(knn, 'static/face_recognition_model.pkl')

def extract_attendance():
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    names = df['Name']
    rolls = df['Roll']
    times = df['Time']
    l = len(df)
    return names, rolls, times, l

def add_attendance(name):
    username = name.split('_')[0]
    userid = name.split('_')[1]
    current_time = datetime.now().strftime("%H:%M:%S")

    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    if int(userid) not in list(df['Roll']):
        with open(f'Attendance/Attendance-{datetoday}.csv', 'a') as f:
            f.write(f'\n{username},{userid},{current_time}')

def getallusers():
    userlist = os.listdir('static/faces')
    names = []
    rolls = []
    l = len(userlist)

    for i in userlist:
        name, roll = i.split('_')
        names.append(name)
        rolls.append(roll)

    return userlist, names, rolls, l

import time

def show_count(frame):
    global image_count, captured_images

    if image_count < 10:  # Capture 10 images
        # Wait for 0.8 second before capturing the next image
        time.sleep(0.1)
        captured_images.append(frame)
        image_count += 1

    # Display the image count on the frame
    cv2.putText(frame, f'Look at camera & Smile please', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Reset the image count to 0 after capturing 10 images
    if image_count == 10:
        image_count = 0

def generate_frames():
    global image_count, captured_images, is_capturing
    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        if not success:
            break

        # Convert the frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        # Draw rectangles around detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Call show_count only if we are in capturing mode
        if is_capturing:
            show_count(frame)  # Update the captured image count

        # Encode the frame in JPEG format
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()


def generate_frames_for_start():
    global image_count, captured_images, is_capturing
    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        if not success:
            break

        # Convert the frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        # Draw rectangles around detected faces and identify the person
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Extract the face for recognition
            face = cv2.resize(frame[y:y + h, x:x + w], (50, 50))
            face = face.reshape(1, -1)

            try:
                # Identify the person
                identified_person = identify_face(face)[0]
                name = identified_person.split('_')[0]

                # Display the name of the person above the face rectangle
                cv2.putText(frame, f'{name}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            except:
                # If the person cannot be identified, show "Unknown"
                cv2.putText(frame, 'Unknown', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Encode the frame in JPEG format
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()



@app.route('/attendance/mark')
def mark_attendance():
    names, rolls, times, l = extract_attendance()  # Extract attendance data
    return render_template('mark_attendance.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)


@app.route('/video_feed')
def video_feed():
    # Return the streaming response using the generator
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video_feed_start')
def video_feed_start():
    # Return the streaming response using the generator
    return Response(generate_frames_for_start(), mimetype='multipart/x-mixed-replace; boundary=frame')



@app.route('/')
def home():
    names, rolls, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)

@app.route('/start', methods=['GET'])
def start():
    names, rolls, times, l = extract_attendance()

    if 'face_recognition_model.pkl' not in os.listdir('static'):
        return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2, mess='There is no trained model in the static folder. Please add a new face to continue.')

    ret = True
    cap = cv2.VideoCapture(0)
    detected_name = None
    while ret:
        ret, frame = cap.read()
        if len(extract_faces(frame)) > 0:
            (x, y, w, h) = extract_faces(frame)[0]
            face = cv2.resize(frame[y:y+h, x:x+w], (50, 50))
            identified_person = identify_face(face.reshape(1, -1))[0]
            add_attendance(identified_person)
            detected_name = identified_person.split('_')[0]
            cv2.putText(frame, f'{detected_name}', (x, y - 15), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
            imgBackground[10:10 + 480, 20:20 + 640] = frame
            cv2.imwrite('static/processed_background.png', imgBackground)
            break
        else:
            imgBackground[10:10 + 480, 20:20 + 640] = frame
            cv2.imwrite('static/processed_background.png', imgBackground)

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    names, rolls, times, l = extract_attendance()
    return render_template('mark_attendance.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2, detected_name=detected_name, background_image='static/processed_background.png')


@app.route('/attendance')
def attendance():
    names, rolls, times, l = extract_attendance()  # Extract attendance data
    return render_template('add_attendance.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)

@app.route('/add', methods=['GET', 'POST'])
def add():
    global image_count, captured_images, is_capturing
    is_capturing = True  # Set capturing flag to True
    newusername = request.form['newusername']
    newuserid = request.form['newuserid']
    userimagefolder = f'static/faces/{newusername}_{str(newuserid)}'
    
    if not os.path.isdir(userimagefolder):
        os.makedirs(userimagefolder)

    i, j = 0, 0
    cap = cv2.VideoCapture(0)
    
    while True:
        _, frame = cap.read()
        faces = extract_faces(frame)
        
        for (x, y, w, h) in faces:
            # Draw rectangle around face and show number of images captured
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 20), 2)
            cv2.putText(frame, f'Images Captured: {i}/{nimgs}', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2, cv2.LINE_AA)

            # Save image every 5 frames, until the required number of images is captured
            if j % 5 == 0 and i < nimgs:  # Only capture if we haven't reached nimgs
                name = newusername + '_' + str(i) + '.jpg'
                cv2.imwrite(f'{userimagefolder}/{name}', frame[y:y + h, x:x + w])
                i += 1

            j += 1
            # Call show_count to update captured images count
            show_count(frame)  # Update image count

        # Exit the loop when we have captured enough images
        if i == nimgs:
            break

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    print('Training Model')

    # Train the model with new images
    train_model()

    # Extract attendance details to pass to the homepage
    names, rolls, times, l = extract_attendance()

    is_capturing = False  # Reset capturing flag to False

    return render_template('add_attendance.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)



if __name__ == '__main__':
    app.run(debug=True)