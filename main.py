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
xServoPin = 13
yServoPin = 12


TRACKING_SENSITIVITY_X = 0.002  # How much to move servo per pixel difference
TRACKING_SENSITIVITY_Y = 0.002
SERVO_SMOOTHING = 0.9  # Smoothing factor (0-1, higher = smoother but slower)


# we will be using gpiozeros servo class to control the servos with PWM
xServo = Servo(xServoPin, min_pulse_width=0.0005, max_pulse_width=0.00245)  # min and max pulse widths are found on the servos datasheet
yServo = Servo(yServoPin, min_pulse_width=0.0005, max_pulse_width=0.00245) 

# Calibration offsets and servo positions used for testing
xOffset = 0
yOffset = 0
# the idea is that we will get the width and height of the camera view and map that to -1 to 1 for both the y and x axis
currentXServoPos = 0  # Range from -1 to 1
currentYServoPos = 0  # Range from -1 to 1

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
        # xServo.value = currentXServoPos
        print(f"move left - X servo: {currentXServoPos}")

    def moveRight(self):
        global currentXServoPos
        currentXServoPos = min(1, currentXServoPos + 0.1)
        # xServo.value = currentXServoPos
        print(f"move right - X servo: {currentXServoPos}")

    def moveUp(self):
        global currentYServoPos
        currentYServoPos = max(-1, currentYServoPos - 0.1)
        # yServo.value = currentYServoPos
        print(f"move up - Y servo: {currentYServoPos}")

    def moveDown(self):
        global currentYServoPos
        currentYServoPos = min(1, currentYServoPos + 0.1)
        # yServo.value = currentYServoPos
        print(f"move down - Y servo: {currentYServoPos}")

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
        # cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) # not requried but may help

        # Print to confirm
        print("Camera settings:")
        print(f"  Width:  {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}")
        print(f"  Height: {cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
        print(f"  FPS:    {cap.get(cv2.CAP_PROP_FPS)}")


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
                        

                        # # convert the error (position of eyes on grid x: 0-width y:0-height) to actual positional arguments -1 to 1
                        # servo_x = ((target_x/w) * 2) - 1
                        # servo_y = ((target_y/h) * 2) - 1

                        
                        # # Smooth the movement
                        # currentXServoPos = servo_x 
                        # currentYServoPos = servo_y
                        
                        # # Clamp to servo limits
                        # currentXServoPos = max(-1, min(1, currentXServoPos))
                        # currentYServoPos = max(-1, min(1, currentYServoPos))
                        
                        # # Move servos (uncomment when servos are connected)
                        # xServo.value = currentXServoPos
                        # yServo.value = currentYServoPos
                        

                        error_x = target_x - frame_center_x
                        error_y = target_y - frame_center_y

                        # Calculate servo adjustments (invert X if needed based on your servo orientation)
                        servo_adjust_x = error_x * TRACKING_SENSITIVITY_X
                        servo_adjust_y = error_y * TRACKING_SENSITIVITY_Y

                        # Apply smoothing and update servo positions
                        new_x_pos = currentXServoPos + servo_adjust_x
                        new_y_pos = currentYServoPos + servo_adjust_y
                        smoothing = 0.7
                        # Smooth the movement
                        currentXServoPos = currentXServoPos * SERVO_SMOOTHING + new_x_pos * (1 - SERVO_SMOOTHING)
                        currentYServoPos = currentYServoPos * SERVO_SMOOTHING + new_y_pos * (1 - SERVO_SMOOTHING)

                        # Clamp to servo limits
                        currentXServoPos = max(-1, min(1, currentXServoPos))
                        currentYServoPos = max(-1, min(1, currentYServoPos))

                        # Move servos
                        xServo.value = currentXServoPos
                        yServo.value = -currentYServoPos

                        # Add tracking status to frame
                        cv2.putText(frame, "TRACKING", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.putText(frame, f"Servo X: {currentXServoPos:.3f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        cv2.putText(frame, f"Servo Y: {currentYServoPos:.3f}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        
                        print(f"Tracking - Eyes: ({eye_center_x}, {eye_center_y}) Servo: ({currentXServoPos}, {currentYServoPos})")

                    # Store last eye position for offset calibration
                    if hasattr(gui, 'lastEyeX'):
                        gui.lastEyeX = eye_center_x
                        gui.lastEyeY = eye_center_y

                else:
                    # No face detected
                    cv2.putText(frame, "NO FACE DETECTED", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # Calculate and display FPS
                # frame_count += 1
                # if frame_count % 30 == 0:  # Update FPS every 30 frames
                #     fps_end_time = time.time()
                #     fps = 30 / (fps_end_time - fps_start_time)
                #     fps_start_time = fps_end_time
                #     print(f"FPS: {fps:.1f}")

                # # Display FPS on frame
                # cv2.putText(frame, f"FPS: {cap.get(cv2.CAP_PROP_FPS):.1f}", (w-150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

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
    # Start tracking in a separate thread
    trackingThread = threading.Thread(target=trackEyes, daemon=True)
    trackingThread.start()
    
    # Initialize and Start GUI in main thread
    gui = Gui()
    gui.startGui()
    
    # Clean up once gui is closed
    print("Program ended")