# these are the imports we will use in this program
import tkinter as tk
import threading
import cv2
import mediapipe as mp
import busio
import board
from adafruit_pca9685 import PCA9685
from adafruit_motor import servo
import csv
import time

# defines global variables we will use
# these are flags to control the application state
quitApplication = False
tracking = False

# PCA9685 servo channel assignments (0-15 available)
X_SERVO_CHANNEL = 0
Y_SERVO_CHANNEL = 1

# Servo configuration
SERVO_MIN_PULSE = 500   # Minimum pulse width in microseconds
SERVO_MAX_PULSE = 2450  # Maximum pulse width in microseconds
SERVO_FREQUENCY = 50    # PWM frequency for servos (50Hz standard)

TRACKING_SENSITIVITY_X = 0.001  # How much to move servo per pixel difference
TRACKING_SENSITIVITY_Y = 0.001
SERVO_SMOOTHING = 0.65  # Smoothing factor (0-1, higher = smoother but slower)

# Initialize I2C bus and PCA9685
i2c = busio.I2C(board.SCL, board.SDA)
pca = PCA9685(i2c)
pca.frequency = SERVO_FREQUENCY

# Create servo objects
xServo = servo.Servo(pca.channels[X_SERVO_CHANNEL], min_pulse=SERVO_MIN_PULSE, max_pulse=SERVO_MAX_PULSE)
yServo = servo.Servo(pca.channels[Y_SERVO_CHANNEL], min_pulse=SERVO_MIN_PULSE, max_pulse=SERVO_MAX_PULSE)

# Calibration offsets and servo positions used for testing
xOffset = 0
yOffset = 0
# Servo positions in degrees (typically 0-180 degrees)
currentXServoPos = 90  # Center position
currentYServoPos = 90  # Center position

# Helper functions for servo control
def setServoAngle(servo_obj, angle):
    """Set servo to specific angle (0-180 degrees)"""
    try:
        # Clamp angle to valid range
        angle = max(0, min(180, angle))
        servo_obj.angle = angle
    except Exception as e:
        print(f"Error setting servo angle: {e}")

def mapToServoAngle(normalized_value):
    """Convert normalized value (-1 to 1) to servo angle (0-180 degrees)"""
    return int(90 + (normalized_value * 90))  # Center at 90 degrees

def centerServos():
    """Center both servos to 90 degrees"""
    global currentXServoPos, currentYServoPos
    currentXServoPos = 90
    currentYServoPos = 90
    setServoAngle(xServo, 90)
    setServoAngle(yServo, 90)
    print("Servos centered at 90 degrees")

# this is the GUI class using tkinter library documentation can be found at https://docs.python.org/3/library/tk.html
class Gui:
    # initializes the gui and sets up layout
    def __init__(self):
        # defines root element which everything else gets added to this element
        # basically the main window and tkinter elements are added to this via the pack method
        self.root = tk.Tk()
        self.root.title("The Awesome Eye Tracker") # defines window title :)

        # the way tkinter works is by first creating an element such as label or button with root as the parent element
        label = tk.Label(self.root, text="Press Start to begin tracking") 
        # then the created elements are packed/ added to the window 
        label.pack(pady=10)

        # Creates buttons and the methods run are the corresponding method names given to command parameter
        # so th method set to command (in this case setStartTracking) will run when the Start button is pressed
        start_button = tk.Button(self.root, text="Start", command=self.setStartTracking)
        start_button.pack(pady=10)

        stop_button = tk.Button(self.root, text="Stop", command=self.stopTracking)
        stop_button.pack(pady=5)

        exit_button = tk.Button(self.root, text="Exit", command=self.endProgram)
        exit_button.pack(pady=10)

        calibrationFrame = tk.Frame(self.root, padx=10, pady=10)
        calibrationFrame.pack(pady=10)

        # creates a smaller frame inside for an arrow control panel for adjusting offsets for testing
        arrowFrame = tk.Frame(calibrationFrame, bg="lightgray", padx=10, pady=10)
        arrowFrame.pack(side=tk.RIGHT, pady=10)

        tk.Button(arrowFrame, text="left", command=self.moveLeft).grid(column=0, row=1)
        tk.Button(arrowFrame, text="right", command=self.moveRight).grid(column=2, row=1)
        tk.Button(arrowFrame, text="up", command=self.moveUp).grid(column=1, row=0)
        tk.Button(arrowFrame, text="down", command=self.moveDown).grid(column=1, row=2)

        # this frame contains the arrow frame and the buttons for settings offsets
        setFrame = tk.Frame(calibrationFrame, padx=10)
        setFrame.pack(side=tk.LEFT)

        tk.Button(setFrame, text="Set Offset For X", command=self.setMaxX).pack(pady=5)
        tk.Button(setFrame, text="Set Offset For Y", command=self.setMaxY).pack(pady=5)
        tk.Button(setFrame, text="Center Servos", command=self.centerServos).pack(pady=5)

    # this sets the tracking flag to true
    def setStartTracking(self):
        global tracking
        tracking = True
        print("start tracking")

    # this sets the tracking flag to false
    def stopTracking(self):
        global tracking
        tracking = False
        print("stop tracking")

    # this sets the quitApplication flag to true and then self.root.quit() ends the GUI loop
    def endProgram(self):
        global quitApplication
        quitApplication = True
        print("ending program")
        # Clean up PCA9685
        pca.deinit()
        self.root.quit() 

    # this method manually moves the servos in the corresponding direction by adjusting angle by 5 degrees
    def moveLeft(self):
        global currentXServoPos
        currentXServoPos = max(0, currentXServoPos - 5)
        setServoAngle(xServo, currentXServoPos)
        print(f"move left - X servo: {currentXServoPos} degrees")

    def moveRight(self):
        global currentXServoPos
        currentXServoPos = min(180, currentXServoPos + 5)
        setServoAngle(xServo, currentXServoPos)
        print(f"move right - X servo: {currentXServoPos} degrees")

    def moveUp(self):
        global currentYServoPos
        currentYServoPos = max(0, currentYServoPos - 5)
        setServoAngle(yServo, currentYServoPos)
        print(f"move up - Y servo: {currentYServoPos} degrees")

    def moveDown(self):
        global currentYServoPos
        currentYServoPos = min(180, currentYServoPos + 5)
        setServoAngle(yServo, currentYServoPos)
        print(f"move down - Y servo: {currentYServoPos} degrees")

    def setMaxX(self):
        global xOffset
        # Store current eye position as X offset
        xOffset = getattr(self, 'lastEyeX', 0)
        print(f"Set X offset to: {xOffset}")

    def setMaxY(self):
        global yOffset
        # Store current eye position as Y offset
        yOffset = getattr(self, 'lastEyeY', 0)
        print(f"Set Y offset to: {yOffset}")

    # this is to center the position of the servos when not tracking
    def centerServos(self):
        centerServos()

    # this method starts the GUI loop IMPORTANT: once this is called the program thread will be locked in the loop
    # this is why its important to start the other thread before calling this method
    def startGui(self):
        self.root.mainloop()

# this method is going to be sent to the second thread to run the eye tracking code and send it to the servos
def trackEyes():
    global currentXServoPos, currentYServoPos # defines the global variables we will use to be global so we can edit them

    startTime = time.time()
    
    with open('data.csv', 'w', newline='') as csvfile:
        fieldnames=['time', 'x', 'y']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # this sets up mediapipe for face mesh detection meaning it will apply a mesh to the detected face landmarks for better targetting
        mp_face_mesh = mp.solutions.face_mesh
        # these are the coordinates of the face mesh for the left and right iris four for the top, bottom, left and right of the iris
        LEFT_IRIS = [474, 475, 476, 477]
        RIGHT_IRIS = [469, 470, 471, 472]

        # Try multiple camera indices and backends for better compatibility
        cap = None
        for camera_index in [0, 1, 2]:
            try:
                print(f"Trying camera index {camera_index}...")
                cap = cv2.VideoCapture(camera_index)
                
                # Test if camera is working by reading a frame
                ret, test_frame = cap.read()
                if ret and test_frame is not None:
                    print(f"Successfully connected to camera {camera_index}")
                    break
                else:
                    cap.release()
                    cap = None
            except Exception as e:
                print(f"Camera index {camera_index} failed: {e}")
                if cap:
                    cap.release()
                cap = None
        
        if cap is None:
            print("ERROR: No working camera found!")
            return
        
        w, h = 640, 480

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Print to confirm
        print("Camera settings:")
        print(f"  Width:  {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}")
        print(f"  Height: {cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
        print(f"  FPS:    {cap.get(cv2.CAP_PROP_FPS)}")

        # Initialize servos to center position
        centerServos()

        # Initialize mediapipe with optimized settings for better performance
        with mp_face_mesh.FaceMesh(
            max_num_faces=1, 
            refine_landmarks=True, 
            min_detection_confidence=0.7,  # Increased for better stability
            min_tracking_confidence=0.5
        ) as face_mesh:
            
            # this is the second looped thread that will run the eye tracking code indefinitely until stopped 
            while cap.isOpened() and not quitApplication:
                
                # Read current frame
                success, frame = cap.read()
                if not success:
                    print("ERROR - Could not read frame from camera")
                    continue  # Try next frame instead of breaking
                
                # Flip frame horizontally for mirror effect (more intuitive)
                frame = cv2.flip(frame, 1)
                
                # Calculate center of frame for reference
                frame_center_x = w // 2
                frame_center_y = h // 2

                # Process every frame but only do heavy processing when needed
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False  # Improve performance
                
                # Run mediapipe face mesh
                results = face_mesh.process(rgb)
                
                # Make frame writeable again for drawing
                rgb.flags.writeable = True

                if results.multi_face_landmarks:
                    # Get the face mesh landmarks
                    mesh = results.multi_face_landmarks[0].landmark

                    # Get iris coordinates for both eyes
                    leftCords = [(int(mesh[p].x * w), int(mesh[p].y * h)) for p in LEFT_IRIS]
                    rightCords = [(int(mesh[p].x * w), int(mesh[p].y * h)) for p in RIGHT_IRIS]

                    # Calculate eye centers
                    left_center = tuple(map(lambda x: sum(x) // len(x), zip(*leftCords)))
                    right_center = tuple(map(lambda x: sum(x) // len(x), zip(*rightCords)))

                    # Average of both eyes for stable tracking
                    eye_center_x = (left_center[0] + right_center[0]) // 2
                    eye_center_y = (left_center[1] + right_center[1]) // 2

                    # Apply offsets
                    target_x = eye_center_x + xOffset
                    target_y = eye_center_y + yOffset

                    # Draw tracking indicators
                    cv2.circle(frame, left_center, 3, (0, 255, 0), -1)
                    cv2.circle(frame, right_center, 3, (0, 255, 0), -1)
                    cv2.circle(frame, (eye_center_x, eye_center_y), 5, (255, 0, 0), 2)
                    cv2.circle(frame, (target_x, target_y), 7, (0, 0, 255), 2)
                    
                    # Draw crosshairs at frame center
                    cv2.line(frame, (frame_center_x - 20, frame_center_y), (frame_center_x + 20, frame_center_y), (255, 255, 255), 2)
                    cv2.line(frame, (frame_center_x, frame_center_y - 20), (frame_center_x, frame_center_y + 20), (255, 255, 255), 2)

                    if tracking:
                        # Log data to CSV
                        writer.writerow({"time": time.time() - startTime, "x": eye_center_x, "y": eye_center_y})
                        csvfile.flush()

                        # Calculate error from center
                        error_x = target_x - frame_center_x
                        error_y = target_y - frame_center_y

                        # Calculate servo adjustments
                        servo_adjust_x = error_x * TRACKING_SENSITIVITY_X * 90  # Scale to degrees
                        servo_adjust_y = error_y * TRACKING_SENSITIVITY_Y * 90

                        # Apply smoothing and update servo positions
                        new_x_pos = currentXServoPos + servo_adjust_x
                        new_y_pos = currentYServoPos + servo_adjust_y

                        # Smooth the movement
                        currentXServoPos = currentXServoPos * SERVO_SMOOTHING + new_x_pos * (1 - SERVO_SMOOTHING)
                        currentYServoPos = currentYServoPos * SERVO_SMOOTHING + new_y_pos * (1 - SERVO_SMOOTHING)

                        # Clamp to servo limits (0-180 degrees)
                        currentXServoPos = max(0, min(180, currentXServoPos))
                        currentYServoPos = max(0, min(180, currentYServoPos))

                        # Move servos
                        setServoAngle(xServo, currentXServoPos)
                        setServoAngle(yServo, currentYServoPos)

                        # Add tracking status to frame
                        cv2.putText(frame, "TRACKING", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.putText(frame, f"Servo X: {currentXServoPos:.1f}°", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        cv2.putText(frame, f"Servo Y: {currentYServoPos:.1f}°", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        
                        print(f"Tracking - Eyes: ({eye_center_x}, {eye_center_y}) Servo: ({currentXServoPos:.1f}°, {currentYServoPos:.1f}°)")

                    # Store last eye position for offset calibration
                    if hasattr(gui, 'lastEyeX'):
                        gui.lastEyeX = eye_center_x
                        gui.lastEyeY = eye_center_y

                else:
                    # No face detected
                    cv2.putText(frame, "NO FACE DETECTED", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # Display the frame
                cv2.imshow("Eye Tracker", frame)
                
                # Check for 'q' key press to quit (optional manual exit)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    # Cleanup
    if cap:
        cap.release()
    cv2.destroyAllWindows()
    print("Camera and windows cleaned up")

# This is the main entry point of the program
if __name__ == "__main__":
    try:
        # Start tracking in a separate thread
        trackingThread = threading.Thread(target=trackEyes, daemon=True)
        trackingThread.start()
        
        # Initialize and Start GUI in main thread
        gui = Gui()
        gui.startGui()
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Clean up PCA9685 on exit
        try:
            pca.deinit()
        except:
            pass
        print("Program ended")