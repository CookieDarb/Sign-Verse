import logging
import sys
from flask import Flask, render_template, Response, request
import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import math
from flask_socketio import SocketIO
global current_model
global toggle
app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0  # Disable caching
socketio = SocketIO(app)

detector = HandDetector(detectionCon=0.8)

offset = 20
img_size = 600
labels_alphabets = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S",
                    "T", "U", "V", "W", "X", "Y", "Z"]
labels_numbers = ["1", "2", "3", "4", "5", "6", "7", "8", "9"]
current_model = "alphabets"

classifier_alphabets = Classifier("Model_Alpha/keras_model.h5", "Model_Alpha/labels.txt")
classifier_numbers = Classifier("Model_Num/keras_model.h5", "Model_Num/labels.txt")

toggle = False

@app.route('/')
@app.route('/index.html')
def index():
    return render_template('index.html')


@app.route('/Sign-to-text.html')
def sign_to_text():
    global current_model 
    current_model= "alphabets"
    return render_template('Sign-to-text.html')


@app.route('/Text-Image.html')
def text_to_sign():
    return render_template('Text-Image.html')


@app.route('/About.html')
def about():
    return render_template('About.html')


@app.route('/Sign.html')
def sign():
    return render_template('Sign.html')


def gen_frames():
    global current_model
    global toggle
    detected_text = ""
    count = 0
    maxi = 0
    listop = list()
    coord_x = coord_y = coord_xf = coord_yf = None
    cap = cv2.VideoCapture(0)
    while True:
        if toggle:
            toggle = False
            break
        imgWhite = np.ones((img_size, img_size, 3), np.uint8) * 255

        success, img = cap.read()
        imgoutput = img.copy()
        hands, img = detector.findHands(img)
        if hands:
            rcnt = 0
            lcnt = 0
            for hand in hands:
                if hand['type'] == 'Left':
                    lcnt = 1
                    left = hand
                    xl, yl, wl, hl = left['bbox']
                if hand['type'] == 'Right':
                    rcnt = 1
                    right = hand
                    xr, yr, wr, hr = right['bbox']
            try:
                if rcnt == 1 and lcnt == 1:
                    imgWhite = np.ones((img_size, img_size, 3), np.uint8) * 255
                    imgCrop = img[min(yl, yr) - offset * 2: max(yl + hl, yr + hr) + max(hl, hr) + offset * 2,
                              xr - offset: xr + wl + wr + offset * 2]
                    imgCropshape = imgCrop.shape

                    w = max(xr - xl + wr, xl - xr + wl)
                    h = max(yr - yl + hr, yl - yr + hl)

                    ar = h / w
                    if ar > 1:
                        k = img_size / min(h, img_size)
                        wcal = min(math.ceil(w * k), img_size)
                        imgResize = cv2.resize(imgCrop, (wcal, img_size))
                        wgap = math.ceil((img_size - wcal) / 2)
                        imgWhite[:, wgap:wgap + wcal] = imgResize
                    else:
                        k = img_size / min(wr + wl, img_size)
                        hcal = min(math.ceil(h * k), img_size)
                        imgResize = cv2.resize(imgCrop, (img_size, hcal))
                        hgap = math.ceil((img_size - hcal) / 2)
                        imgWhite[hgap:hgap + hcal, :] = imgResize
                    coord_x, coord_y = xr - offset, min(yl, yr) - offset * 2
                    coord_xf, coord_yf = xr + wl + wr + offset * 2, max(yl + hl, yr + hr) + max(hl, hr) + offset * 2
                elif rcnt == 1 and lcnt == 0:
                    imgCrop = img[yr - offset: yr + hr + offset, xr - offset: xr + wr + offset]
                    ar = hr / wr
                    if ar > 1:
                        k = img_size / hr
                        wcal = math.ceil(wr * k)
                        imgResize = cv2.resize(imgCrop, (wcal, img_size))
                        wgap = math.ceil((img_size - wcal) / 2)
                        imgWhite[:, wgap:wgap + wcal] = imgResize
                    else:
                        k = img_size / wr
                        hcal = math.ceil(hr * k)
                        imgResize = cv2.resize(imgCrop, (img_size, hcal))
                        hgap = math.ceil((img_size - hcal) / 2)
                        imgWhite[hgap:hgap + hcal, :] = imgResize
                    coord_x, coord_y = xr - offset, yr - offset
                    coord_xf, coord_yf = xr + wr + offset, yr + hr + offset
                elif rcnt == 0 and lcnt == 1:
                    imgCrop = img[yl - offset: yl + hl + offset, xl - offset: xl + wl + offset]
                    ar = hl / wl
                    if ar > 1:
                        k = img_size / hl
                        wcal = math.ceil(wl * k)
                        imgResize = cv2.resize(imgCrop, (wcal, img_size))
                        wgap = math.ceil((img_size - wcal) / 2)
                        imgWhite[:, wgap:wgap + wcal] = imgResize
                    else:
                        k = img_size / wl
                        hcal = math.ceil(hl * k)
                        imgResize = cv2.resize(imgCrop, (img_size, hcal))
                        hgap = math.ceil((img_size - hcal) / 2)
                        imgWhite[hgap:hgap + hcal, :] = imgResize
                    coord_x, coord_y = xl - offset, yl - offset
                    coord_xf, coord_yf = xl + wl + offset, yl + hl + offset
            except Exception as e:
                pass

            if current_model == "alphabets":
                classifier = classifier_alphabets
                labels = labels_alphabets
            else:
                classifier = classifier_numbers
                labels = labels_numbers

            original_stdout = sys.stdout
            sys.stdout = open('dummy', 'w')
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            sys.stdout.close()
            sys.stdout = original_stdout

            if index is not None:
                if count <= 20:
                    count += 1
                    listop.append(index)
                else:
                    count = 0
                    maxc, maxi = 0, 0
                    indexdone = list()
                    for i in listop:
                        indexc = 0
                        if i in indexdone:
                            continue
                        for j in listop:
                            if i == j:
                                indexc += 1
                        if indexc > maxc:
                            maxc = indexc
                            maxi = i
                        indexdone.append(i)
                    detected_text = labels[maxi]
                    listop = list()

        if coord_x and coord_y and coord_xf and coord_yf:
            cv2.putText(imgoutput, labels[maxi], (coord_x, coord_y - 20), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 255),
                        2)
            cv2.rectangle(imgoutput, (coord_x, coord_y), (coord_xf, coord_yf), (255, 0, 255), 4)

        ret, frame = cv2.imencode('.jpg', imgoutput)
        frame_bytes = frame.tobytes()
        socketio.emit('detected_text', detected_text)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@socketio.on('toggle_signs')
def handle_toggle_signs(sign_type):
    global current_model
    
    current_model = sign_type
    socketio.emit('model_changed', current_model)



@app.route('/stop_feed', methods=['POST'])
def stop_feed():
    global toggle
    toggle = True
    return 'Feed stopped'
    


if __name__ == '__main__':
    socketio.run(app, debug=True)
