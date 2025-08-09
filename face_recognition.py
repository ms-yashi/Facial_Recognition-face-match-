import cv2
import numpy as np
import os

# Path to Haar cascade file for face detection
haar_file = 'haarcascade_frontalface_default.xml'

# Path to training dataset
datasets = 'datasets'

print('Training...')

# Lists for training data
(images, labels, names, id) = ([], [], {}, 0)

# Load all images and labels from dataset folders
(width, height) = (130, 100)  # Define standard size

for (subdirs, dirs, files) in os.walk(datasets):
    for subdir in dirs:
        names[id] = subdir
        subjectpath = os.path.join(datasets, subdir)
        for filename in os.listdir(subjectpath):
            path = os.path.join(subjectpath, filename)
            img = cv2.imread(path, 0)  # Read image in grayscale
            img = cv2.resize(img, (width, height))  # ✅ Resize image
            images.append(img)
            labels.append(id)
        id += 1


# Resize target image dimensions
(width, height) = (130, 100)

# Convert lists to NumPy arrays
(images, labels) = [np.array(lis) for lis in [images, labels]]

# Create and train the model
model = cv2.face.FisherFaceRecognizer_create()
model.train(images, labels)

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(haar_file)

# Start webcam
webcam = cv2.VideoCapture(0)
cnt = 0

# Recognition loop
while True:
    ret, im = webcam.read()
    if not ret:
        break

    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = gray[y:y + h, x:x + w]
        face_resize = cv2.resize(face, (width, height))

        prediction = model.predict(face_resize)
        cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 3)

        if prediction[1] < 800:
            name = names[prediction[0]]
            confidence = int(prediction[1])
            cv2.putText(im, f'{name} - {confidence}', (x - 10, y - 10),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (51, 255, 255), 2)
            print(name)
            cnt = 0
        else:
            cnt += 1
            cv2.putText(im, 'Unknown', (x - 10, y - 10),
                        cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
            if cnt > 100:
                print("Unknown Person")
                cv2.imwrite("input.jpg", im)
                cnt = 0

    cv2.imshow('Face Recognition', im)
    key = cv2.waitKey(10)
    if key == 27:  # ESC key to exit
        break

webcam.release()
cv2.destroyAllWindows()  


