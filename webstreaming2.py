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
from multiprocessing import Pool

url = "http://10.0.0.20:8000/movement/"

files = []
headers = {
  'Authorization': 'Token 289c530b43bb6fdd984e02f562037d18a759d6c3'
}

app = Flask(__name__)

vs = VideoStream(usePiCamera=1).start()
time.sleep(2.0)

def detect_motion_post(motion_interval):
    print(motion_interval)
    payload = {'baby': '1', 'start_time': motion_interval[0], 'end_time': motion_interval[1]}
    response = requests.request("POST", url, headers=headers, data = payload, files = files)
    print(response.text.encode('utf8'))


def generate():

    # Init motion detector and total frames read thus far
    md = SingleMotionDetector(accumWeight=0.1)
    total = 0
    frame_count = 32

    pool = Pool(processes=3)
    motion_interval = [None, None]
    still_start = None
    while True:

        frame = vs.read()
        datetime_now = datetime.datetime.now()

        frame = imutils.resize(frame, width=400)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)
        
        # if the total number of frames has reached a sufficient
        # number to construct a reasonable background model, then
        # continue to process the frame
        if total > frame_count:
            # detect motion in the image
            motion = md.detect(gray)

            if motion is None and still_start == None:
                still_start = datetime_now
            elif motion is not None:
                still_start = None

            # check to see if motion was found in the frame
            if motion is not None and motion_interval[0] == None:
                print('start')
                motion_interval[0] = datetime_now.time()

            elif motion is not None and motion_interval[0] != None:
                continue

            elif (datetime_now - still_start).total_seconds() > 5 and motion_interval[0] != None:
                print('end')
                motion_interval[1] = datetime_now.time()
                pool.apply_async(detect_motion_post, (motion_interval, ))
                motion_interval = [None, None]

        
        # update the background model and increment the total number
        # of frames read thus far
        md.update(gray)
        total += 1
            
        # encode the frame in JPEG format
        (flag, encodedImage) = cv2.imencode(".jpg", frame)
        # ensure the frame was successfully encoded
        if not flag:
            continue

        # yield the output frame in the byte format
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
            bytearray(encodedImage) + b'\r\n')


@app.route("/")
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

    # start the flask app
    app.run(host=args["ip"], port=args["port"], debug=True,
        threaded=True, use_reloader=False)

    motion_process.join()
# release the video stream pointer
vs.stop()

