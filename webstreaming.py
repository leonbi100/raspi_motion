from motion_detection.motion_detector import SingleMotionDetector
from imutils.video import VideoStream
from flask import Response
from flask import Flask
from flask import render_template
import threading
import argparse
import datetime
import imutils
import time
import cv2
import requests
import queue

url = "http://10.0.0.20:8000/movement/"


files = []
headers = {
  'Authorization': 'Token 091491a727c3f0a95cc2089bda2e4a19c88da3af'
}


movements = queue.Queue()
cv = threading.Condition()

app = Flask(__name__)

vs = VideoStream(usePiCamera=1).start()
time.sleep(2.0)

@app.route("/")
def index():
    return render_template("index.html")

def detect_motion(frame_count):
    global vs, lock

    # Init motion detector and total frames read thus far
    md = SingleMotionDetector(accumWeight=0.1)
    total = 0

    still_count = 0
    motion_interval = [None, None]

    while True:
        # read the next frame from the video stream, resize it,
        # convert the frame to grayscale, and blur it
        frame = vs.read()
        frame = imutils.resize(frame, width=400)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)
        
        # if the total number of frames has reached a sufficient
        # number to construct a reasonable background model, then
        # continue to process the frame
        if total > frame_count:
            # detect motion in the image
            motion = md.detect(gray)

            if motion is None:
                still_count += 1
            else:
                still_count = 0

            # check to see if motion was found in the frame
            if motion is not None and motion_interval[0] == None:
                motion_interval[0] = datetime.datetime.now().time()

            if still_count > 100 and motion_interval[0] != None:
                motion_interval[1] = datetime.datetime.now().time()

                cv.acquire()
                movements.put(motion_interval)
                cv.notify()
                cv.release()

                motion_interval = [None, None]

        
        # update the background model and increment the total number
        # of frames read thus far
        md.update(gray)
        total += 1

def generate():

    while True:

        frame = vs.read()
            
        # encode the frame in JPEG format
        (flag, encodedImage) = cv2.imencode(".jpg", frame)
        # ensure the frame was successfully encoded
        if not flag:
            continue

        # yield the output frame in the byte format
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
            bytearray(encodedImage) + b'\r\n')

def post_data():

    while True:
        cv.acquire()
        while movements.empty():
            cv.wait()

        motion_interval = movements.get()
        payload = {'baby': '1', 'start_time': motion_interval[0], 'end_time': motion_interval[1]}

        response = requests.request("POST", url, headers=headers, data = payload, files = files)
        print(response.text.encode('utf8'))

        cv.release()


@app.route("/video_feed")
def video_feed():
    # return the response generated along with the specific media
    # type (mime type)
    return Response(generate(),
        mimetype = "multipart/x-mixed-replace; boundary=frame")


if __name__ == '__main__':
    # construct the argument parser and parse command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--ip", type=str, required=True,
        help="ip address of the device")
    ap.add_argument("-o", "--port", type=int, required=True,
        help="ephemeral port number of the server (1024 to 65535)")
    ap.add_argument("-f", "--frame-count", type=int, default=32,
        help="# of frames used to construct the background model")
    args = vars(ap.parse_args())

    # start a thread that will perform motion detection
    motion_thread = threading.Thread(target=detect_motion, args=(
        args["frame_count"],), daemon=True)
    motion_thread.start()

    post_thread = threading.Thread(target=post_data, daemon=True) 
    post_thread.start()

    # start the flask app
    app.run(host=args["ip"], port=args["port"], debug=True,
        threaded=True, use_reloader=False)
# release the video stream pointer
vs.stop()

