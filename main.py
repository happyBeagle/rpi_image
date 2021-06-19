#Modified by smartbuilds.io
#Date: 27.09.20
#Desc: This web application serves a motion JPEG stream
# main.py
# import the necessary packages
from flask import Flask, render_template, Response, request,  jsonify
from model import Inference
from camera import Camera
import numpy as np
import time
import threading
import os 

pi_camera = Camera(flip=False) # flip pi camera if upside down.
model = Inference(model_type = "GTSNet", model_path="/home/pi/models/GTSNet.pth")

# App Globals (do not edit)
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html') #you can customze index.html here

def gen(camera):
    #get camera frame
    while True:
        frame, jpeg = camera.get_frame()
    
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(pi_camera),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


def capture(camera):
    now, jpeg= camera.get_frame()
    global infer_img
    
    infer_img = jpeg

    yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + now + b'\r\n\r\n')


@app.route('/answer')
def answer():
    _, jpeg = pi_camera.get_frame()
    label, time = model.inference(jpeg)   

    ans = {
        "label":label,
        "time":round(time, 2)
    }

    response = jsonify(ans)
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

@app.route('/inference')
def inference():
    return Response(capture(pi_camera), mimetype='multipart/x-mixed-replace; boundary=frame')
    
if __name__ == '__main__':

    app.run(host='0.0.0.0', debug=False)
    


