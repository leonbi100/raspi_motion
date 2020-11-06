import os
import cv2
import multiprocessing as mp
import time
from datetime import datetime
from motion_detection.motion_detector import SingleMotionDetector

class VideoBuffer:
    def __init__(self, fps=None, video_path='videos', buffer_seconds=5):
        self.video_path = os.path.join(os.getcwd(), video_path)
        self.running = mp.Event()
        self.fps = fps
        self.buffer_seconds = buffer_seconds
        self.frame_width = None
        self.frame_height = None
        self.motion_interval = [None, None]
        self.out = None

        if not os.path.exists(self.video_path):
            os.makedirs(self.video_path)        

    def _calculate_fps(self, cap):
        num_frames = 120
        start = time.time()
        for i in range(0, num_frames) :
            ret, frame = cap.read()
        end = time.time()
        seconds = end - start
        self.fps = num_frames / seconds
        return self.fps
    
    def stop(self):
        self.running.wait()
        self.running.clear()

    def run_capture(self, display_video=False):
        print('Booting up video capture...')
        cap = cv2.VideoCapture(0)
        if self.fps == None:
            self._calculate_fps(cap)
        self.frame_width = int(cap.get(3))
        self.frame_height = int(cap.get(4))

        # Init motion detector and total frames read thus far
        md = SingleMotionDetector(accumWeight=0.1)

        self.running.set()
        total = 0
        while self.running.is_set():
            ret, frame = cap.read()
            if ret == True:

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (7, 7), 0)
                if total > 30:  # Need to establish enough background frames first
                    motion = md.detect(gray)
                    
                    if motion is not None:                    
                        # First detection of motion
                        if self.motion_interval[0] == None:
                            # Save video with current timestamp
                            cur_timestamp = datetime.now().timestamp()
                            video_file = os.path.join(self.video_path, f'{cur_timestamp}.avi')
                            self.out = cv2.VideoWriter(video_file, cv2.VideoWriter_fourcc('M','J','P','G'), self.fps, (self.frame_width, self.frame_height))

                            # Record timestamp of motion beginning
                            self.motion_interval[0] = datetime.now().timestamp()
                        
                        self.out.write(frame)

                        # Draw rectangle around motion
                        (thresh, (minX, minY, maxX, maxY)) = motion
                        cv2.rectangle(frame, (minX, minY), (maxX, maxY), (0, 0, 255), 2)
                    
                    # Motion has ended
                    elif motion is None and self.motion_interval[0] != None:
                        # Output motion interval timestamps
                        self.motion_interval[1] = datetime.now().timestamp()
                        print(self.motion_interval)
                        self.motion_interval = [None, None]

                md.update(gray)
                
                if display_video: 
                    cv2.imshow('', frame)

                total += 1

                # Kill the program with key press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    vb = VideoBuffer()
    proc = mp.Process(target=vb.run_capture, args=(True,))
    proc.start()
    time.sleep(60)
    vb.stop()
    proc.join()