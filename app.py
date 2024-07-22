from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import time

app = Flask(__name__)

model = load_model(r'Model\fer13c.keras')


emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/start')    
def start_emotion_detection():
    language = request.args.get('language')
    
    cap = cv2.VideoCapture(0)
    start_time = time.time()
    timeout = 60  
    green = (0, 255, 0)
    
    current_emotion = None
    while True:
        ret, frame = cap.read()
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 6)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), green, 2)
            face_frame = frame[y:y+h, x:x+w]
            gray_face = cv2.cvtColor(face_frame, cv2.COLOR_BGR2GRAY)
            resized_face = cv2.resize(gray_face, (48, 48))
            img_tensor = image.img_to_array(resized_face)
            img_tensor = img_tensor / 255.0
            img_tensor = np.expand_dims(img_tensor, axis=0)

            prediction = model.predict(img_tensor)[0]
            predicted_class_index = np.argmax(prediction)
            current_emotion = emotions[predicted_class_index]

            text_y = y + h + 20
            cv2.putText(frame, current_emotion, (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, green, 2)

        cv2.imshow('Emotion Detection', frame)
        if cv2.waitKey(1) & cv2.waitKey(1) == 13 or (time.time() - start_time) > timeout:
            break
    cap.release()
    cv2.destroyAllWindows()

    url = f'http://127.0.0.1:5000/playlist?t1={language}_{current_emotion}'
    return jsonify(url=url)

@app.route('/playlist')
def playlist():
    return render_template('playlist.html')

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)
