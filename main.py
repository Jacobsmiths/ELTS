# these are the imports we will use in this program
import tkinter as tk
import threading
import cv2
import mediapipe as mp
from gpiozero import Servo
import csv
import time

# defines global variables we will use
# these are flags to control the application state
quitApplication = False
tracking = False

# these are the GPIO pins for the servos (To do: change these to your actual GPIO pins)
xServoPin = 27
yServoPin = 17

# we will be using gpiozeros servo class to control the servos with PWM
# xServo = Servo(xServoPin, min_pulse_widthmin_pulse_width=0.0005, max_pulse_width=0.0025)  # min and max pulse widths are found on the servos datasheet
# yServo = Servo(yServoPin, min_pulse_widthmin_pulse_width=0.0005, max_pulse_width=0.0025) 

# Calibration offsets and servo positions used for testing
xOffset = 0
yOffset = 0
currentXServoPos = 0  # Range from -1 to 1
currentYServoPos = 0  # Range from -1 to 1

# Tracking sensitivity (adjust these values to fine-tune responsiveness)
# used in a make shift PID controller essentially for improved tracking
TRACKING_SENSITIVITY_X = 0.002  # How much to move servo per pixel difference
TRACKING_SENSITIVITY_Y = 0.002
SERVO_SMOOTHING = 0.7  # Smoothing factor (0-1, higher = smoother but slower)

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
        self.root.quit() 

    # this method manually moves the servos in the corresponding direction by setting the servo value to 
    # the current position plus or minus .1 value
    def moveLeft(self):
        global currentXServoPos
        currentXServoPos = max(-1, currentXServoPos - 0.1)
        xServo.value = currentXServoPos
        print(f"move left - X servo: {currentXServoPos}")

    def moveRight(self):
        global currentXServoPos
        currentXServoPos = min(1, currentXServoPos + 0.1)
        xServo.value = currentXServoPos
        print(f"move right - X servo: {currentXServoPos}")

    def moveUp(self):
        global currentYServoPos
        currentYServoPos = max(-1, currentYServoPos - 0.1)
        yServo.value = currentYServoPos
        print(f"move up - Y servo: {currentYServoPos}")

    def moveDown(self):
        global currentYServoPos
        currentYServoPos = min(1, currentYServoPos + 0.1)
        yServo.value = currentYServoPos
        print(f"move down - Y servo: {currentYServoPos}")

    # this doesn't rly fit in the context I will have to do some rework on this
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
        global currentXServoPos, currentYServoPos
        currentXServoPos = 0
        currentYServoPos = 0
        xServo.mid()
        yServo.mid()
        print("Servos centered")

    # this method starts the GUI loop IMPORTANT: once this is called the program thread will be locked in the loop
    # this is why its important to start the other thread before calling this method
    def startGui(self):
        self.root.mainloop()

# this method is going to be sent to the second thread to run the eye tracking code and send it to the servos over GPIO
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

        # this tries to initialize the camera capture (Important: if you are having issues try changing capture port from 0 to 1)
        try:  
            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2000) # this just sets the width and the height of the video output frame 
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
        except Exception as e:
            print(f"Capture is not found: {e}")
            return

        # Center the servos initially before tracking starts
        # xServo.mid()
        # yServo.mid()

        
        # gets the values for the height and width of the frame
        h, w, _ = cap.read()[1].shape
        print(f"Frame size: {w}x{h}")

        # Calculate center of frame for reference
        frame_center_x = w // 2
        frame_center_y = h // 2
        
        # Initialize mediapipe as a context with some parameters, refined landmarks just means it will have more accurate landmarks 
        # and min_detection_confidence is the minimum confidence for detection to be considered valid 
        with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=.5, min_tracking_confidence=0.5) as face_mesh:

            # this is the second looped thread that will run the eye tracking code indefinetly until stopped 
            # basically saying while the camera feed is on and the quitApplication flag is not set
            while cap.isOpened() and not quitApplication:

                # Read current frame, if not break the loop (for when camera gets disconnected)
                success, frame = cap.read()
                if not success:
                    print("ERROR - Could not read capture")
                    break

                # This converts the frame from BGR to RGB for mediapipe processing
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # This runs the current frmae through mediapipe face mesh model
                results = face_mesh.process(rgb)

                if results.multi_face_landmarks: # if face detected
                    # get the results of the face mesh in mesh
                    mesh = results.multi_face_landmarks[0].landmark

                    # Gets the 4 pixel coordinates (top left right and bottom) of the left and right eyes
                    # breaking this down: p for p in _IRIS basically loops through the cords top right left bottom of iris
                    # mesh[].x / mesh[].y gives us coordinates from 0-1  relative to the screen
                    # meaning the screen is the first quadrant. We apply scale of frame width and height to get pixel coordinates
                    leftCords = [(int(mesh[p].x * w), int(mesh[p].y * h)) for p in LEFT_IRIS]
                    rightCords = [(int(mesh[p].x * w), int(mesh[p].y * h)) for p in RIGHT_IRIS]

                    # Calculate the center of the left and right eyes/ finds the middle of the 4 points for each eye
                    left_center = tuple(map(lambda x: sum(x) // len(x), zip(*leftCords)))
                    right_center = tuple(map(lambda x: sum(x) // len(x), zip(*rightCords)))

                    # Use the average of both eyes for more stable tracking / between the two eyes
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
                    
                    # Draw crosshairs at frame center (for reference)
                    cv2.line(frame, (frame_center_x - 20, frame_center_y), (frame_center_x + 20, frame_center_y), (255, 255, 255), 1)
                    cv2.line(frame, (frame_center_x, frame_center_y - 20), (frame_center_x, frame_center_y + 20), (255, 255, 255), 1)

                    if tracking:
                        writer.writerow({"time":time.time()-startTime, "x": eye_center_x,"y": eye_center_y})
                        csvfile.flush()
                        # Calculate error from center
                        error_x = target_x - frame_center_x
                        error_y = target_y - frame_center_y
                        
                        # Calculate servo adjustments (invert X if needed based on your servo orientation)
                        servo_adjust_x = error_x * TRACKING_SENSITIVITY_X
                        servo_adjust_y = error_y * TRACKING_SENSITIVITY_Y
                        
                        # Apply smoothing and update servo positions
                        new_x_pos = currentXServoPos + servo_adjust_x
                        new_y_pos = currentYServoPos + servo_adjust_y
                        
                        # Smooth the movement
                        currentXServoPos = currentXServoPos * SERVO_SMOOTHING + new_x_pos * (1 - SERVO_SMOOTHING)
                        currentYServoPos = currentYServoPos * SERVO_SMOOTHING + new_y_pos * (1 - SERVO_SMOOTHING)
                        
                        # Clamp to servo limits
                        currentXServoPos = max(-1, min(1, currentXServoPos))
                        currentYServoPos = max(-1, min(1, currentYServoPos))
                        
                        # Move servos
                        # xServo.value = currentXServoPos
                        # yServo.value = currentYServoPos
                        
                        print(f"Tracking - Eyes: ({eye_center_x}, {eye_center_y}) Error: ({error_x}, {error_y}) Servo: ({currentXServoPos:.3f}, {currentYServoPos:.3f})")

                    # Store last eye position for offset calibration
                    gui.lastEyeX = eye_center_x
                    gui.lastEyeY = eye_center_y

                # This displays current frame from camera
                cv2.imshow("Eye Tracker", frame)
            

    cap.release()
    cv2.destroyAllWindows()

# This is the main entry point of the program it asks if this is was the page spawned if yes run the code 
if __name__ == "__main__":
    # Start tracking in a separate thread
    # this creates a thread to run the method trackEyes and daemon=True means it will stop if main thread stops (i.e. GUI is closed)
    trackingThread = threading.Thread(target=trackEyes, daemon=True)
    trackingThread.start() # must say start to start the thread running the method
    
    # Initialize and Start GUI in main thread
    gui = Gui()
    gui.startGui()
    
    # Clean up once gui is closed just some extra code if needed
    # quitApplication = True
    # trackingThread.join(timeout=1.0)