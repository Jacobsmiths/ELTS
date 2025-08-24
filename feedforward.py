import math
import threading
import cv2
import mediapipe as mp
import busio
import board
from adafruit_pca9685 import PCA9685
from adafruit_motor import servo
import csv
import time
from GUI import GUI

class ELTS():
    # Define global variables
    quitApplication = False
    tracking = False

    # PCA9685 servo channel assignments (0-15 available)
    X_SERVO_CHANNEL = 0
    Y_SERVO_CHANNEL = 1

    # Servo configuration
    SERVO_MIN_PULSE = 500   # Minimum pulse width in microseconds
    SERVO_MAX_PULSE = 2450  # Maximum pulse width in microseconds
    SERVO_FREQUENCY = 50    # PWM frequency for servos (50Hz standard)
    
    # Servo limits and constraints
    SERVO_MIN_ANGLE = 0
    SERVO_MAX_ANGLE = 180
    SERVO_CENTER_ANGLE = 90

    # Frame dimensions for setpoint calculation you can change this for better resolution im pretty sure webcam geos to 4k
    FRAME_WIDTH = 640
    FRAME_HEIGHT = 480
    FRAME_CENTER_X = FRAME_WIDTH // 2
    FRAME_CENTER_Y = FRAME_HEIGHT // 2

    # these are hard coded points for the irises on the mesh
    LEFT_IRIS = [474, 475, 476, 477]
    RIGHT_IRIS = [469, 470, 471, 472]

    # these are the camera fov angles
    diag_rad = math.radians(90)
    half_diag = math.tan(diag_rad / 2)

    norm = math.sqrt(16**2 + 9**2)

    HORIZONTAL_VIEW = 2 * math.atan((16 / norm) * half_diag)
    VERTICAL_VIEW = 2 * math.atan((9 / norm) * half_diag)


    def __init__(self):
        # Initialize I2C bus an d PCA9685
        i2c = busio.I2C(board.SCL, board.SDA)
        pca = PCA9685(i2c)
        pca.frequency = self.SERVO_FREQUENCY

        # Create servo objects
        self.xServo = servo.Servo(pca.channels[self.X_SERVO_CHANNEL], min_pulse=self.SERVO_MIN_PULSE, max_pulse=self.SERVO_MAX_PULSE)
        self.yServo = servo.Servo(pca.channels[self.Y_SERVO_CHANNEL], min_pulse=self.SERVO_MIN_PULSE, max_pulse=self.SERVO_MAX_PULSE)
        
        # Calibration offsets
        self.xOffset = 0
        self.yOffset = 0

        self.xServo.angle = self.SERVO_CENTER_ANGLE
        self.yServo.angle = self.SERVO_CENTER_ANGLE
        
        self.currTime = time.perf_counter() # TODO might want to move this into the actual phase of tracking instead of having it count up not being

    def initCamera(self):
        # Try the standard indecies for camera ports or whatever
        cap = None
        for camera_index in [0, 1, 2]: 
            try:
                print(f"Trying camera index {camera_index}...")
                cap = cv2.VideoCapture(camera_index)
                # Camera settings
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.FRAME_WIDTH)  # Reduced resolution for better performance
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.FRAME_HEIGHT)
                cap.set(cv2.CAP_PROP_FPS, 60)  # Set FPS
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer size to avoid lag

                # Test if camera is working by reading a frame
                ret, test_frame = cap.read()
                if ret and test_frame is not None:
                    print(f"Successfully connected to camera {camera_index}")
                    break
                else:
                    cap.release()
                    cap = None
                    print(f"failed reading input from camera {camera_index}")
                
            except Exception as e:
                print(f"Camera index {camera_index} failed: {e}")
                if cap:
                    cap.release()
                cap = None
        
        if cap is None:
            print("ERROR: No working camera found!")
            return

        return cap
        
    def initCSV(self, csvfile):
        fieldnames=['time', 'x', 'y']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        return writer
        
    def trackEyes(self, frame, face_mesh):
        """Process a single frame and return eye coordinates"""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            # Get the face mesh landmarks
            mesh = results.multi_face_landmarks[0].landmark

            # using cords from face mesh its converted into cords onto the frame
            frame_h, frame_w, _ = frame.shape
            leftCords = [(int(mesh[p].x * frame_w), int(mesh[p].y * frame_h)) for p in self.LEFT_IRIS]
            rightCords = [(int(mesh[p].x * frame_w), int(mesh[p].y * frame_h)) for p in self.RIGHT_IRIS]

            # Calculate eye centers
            left_center = tuple(map(lambda x: sum(x) // len(x), zip(*leftCords)))
            right_center = tuple(map(lambda x: sum(x) // len(x), zip(*rightCords)))

            # Average of both eyes for middle of face tracking
            eye_center_x = (left_center[0] + right_center[0]) // 2
            eye_center_y = (left_center[1] + right_center[1]) // 2

            # Apply calibration offsets
            xCords = eye_center_x + self.xOffset
            yCords = eye_center_y + self.yOffset

            return (xCords, yCords)
        return None
    
    def calculate(self, xCords, yCords): # so you get the coordinates of the target in the x and y direction
        # normalize the cords by dividing by height and width of the screen respectively
        xNorm = xCords/self.FRAME_WIDTH
        yNorm = yCords/self.FRAME_HEIGHT

        # then multiply by the FOV in degrees of the camera to get the angle required for the servos
        xAngle = xNorm * self.HORIZONTAL_VIEW
        yAngle = yNorm * self.VERTICAL_VIEW
        return (xAngle, yAngle)
    
    def move(self, xAngle, yAngle):
        if(xAngle > 0 and xAngle < self.SERVO_MAX_ANGLE):
            self.xServo.angle = xAngle
        if(yAngle > 0 and yAngle < self.SERVO_MAX_ANGLE):   
            self.yServo.angle = yAngle

    def main(self):
        cap = self.initCamera()
        if cap is None:
            print("Failed to initialize camera")
            return
        
        # Initialize MediaPipe ONCE outside the loop
        face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1, 
            refine_landmarks=True,
            static_image_mode=False,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.2
        )
        with open('data.csv', 'w', newline='') as csvfile:
            writer = self.initCSV(csvfile)
            startTime = time.time()
            try:
                while not self.quitApplication:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame = cv2.flip(frame, 1)
                    result = self.trackEyes(frame, face_mesh)
                    if result is None:
                        # cv2.putText(frame, "NO FACE DETECTED", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2) # this will never show cuz your not displaying frame
                        cv2.putText(frame, f"Servo X: {self.xServo.angle}°", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        cv2.putText(frame, f"Servo Y: {self.yServo.angle}°", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        cv2.imshow("Eye Tracker", frame) # actually shows the frame
                        cv2.waitKey(1)
                        continue  # skip this frame if no face is detected
                    xCords, yCords = result

                    # Draws tracking indicators on target, crosshairs, and servo angle
                    cv2.circle(frame, (xCords, yCords), 5, (255, 0, 0), 2)
                    cv2.line(frame, (self.FRAME_CENTER_X - 20, self.FRAME_CENTER_Y), (self.FRAME_CENTER_X + 20, self.FRAME_CENTER_Y), (255, 255, 255), 2)
                    cv2.line(frame, (self.FRAME_CENTER_X, self.FRAME_CENTER_Y - 20), (self.FRAME_CENTER_X, self.FRAME_CENTER_Y + 20), (255, 255, 255), 2)
                    cv2.putText(frame, f"Servo X: {self.xServo.angle}°", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(frame, f"Servo Y: {self.yServo.angle}°", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.imshow("Eye Tracker", frame) # actually shows the frame
                    cv2.waitKey(1)

                    # logic handling if we are tracking:
                    if self.tracking:
                        # writing data to the csv
                        writer.writerow({"time": time.time() - startTime, "x": xCords, "y": yCords})
                        csvfile.flush()
                        # updates the positions of the servos
                        xAngle, yAngle = self.calculate(xCords=xCords, yCords=yCords)
                        self.move(xAngle,yAngle)
                        
            except Exception as e:
                print(f"Error in tracking loop: {e}")
            finally:
                # Cleanup
                face_mesh.close()
                if cap:
                    cap.release()
                cv2.destroyAllWindows()
                print("Camera and windows cleaned up")

        # clean up for the windows
        if cap:
            cap.release()
        cv2.destroyAllWindows()
        print("Camera and windows cleaned up")

    def startTracking(self):
        self.tracking = True
        print("start tracking")

    def stopTracking(self):
        self.tracking = False
        print("stop tracking")


if __name__ == "__main__":
    # Start tracking in a separate thread\
    elts = ELTS()
    trackingThread = threading.Thread(target=elts.main, daemon=True)
    trackingThread.start()
    
    # Initialize and Start GUI in main thread
    gui = GUI(startCommand=elts.startTracking, stopCommand=elts.stopTracking)
    gui.start()
    
    # Clean up once gui is closed
    print("Program ended")