# these are the imports we will use in this program
import tkinter as tk
import threading
import cv2
import mediapipe as mp
import RPi.GPIO as GPIO
import csv
import time

# defines global variables we will use
# these are flags to control the application state
quitApplication = False
tracking = False

# Hardware PWM pins
X_SERVO_PIN = 12  # Hardware PWM channel 0
Y_SERVO_PIN = 13  # Hardware PWM channel 1

TRACKING_SENSITIVITY_X = 0.002  # How much to move servo per pixel difference
TRACKING_SENSITIVITY_Y = 0.002
SERVO_SMOOTHING = 0.9  # Smoothing factor (0-1, higher = smoother but slower)

# Initialize GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setup(X_SERVO_PIN, GPIO.OUT)
GPIO.setup(Y_SERVO_PIN, GPIO.OUT)

# Create hardware PWM objects (50Hz for servos)
x_servo_pwm = GPIO.PWM(X_SERVO_PIN, 50)
y_servo_pwm = GPIO.PWM(Y_SERVO_PIN, 50)

# Start PWM with center position (7.5% duty cycle = 1.5ms pulse)
x_servo_pwm.start(7.5)
y_servo_pwm.start(7.5)

# Calibration offsets and servo positions used for testing
xOffset = 0
yOffset = 0
# Servo positions in duty cycle percentage (5-10% for most servos)
currentXServoDuty = 7.5  # Center position
currentYServoDuty = 7.5  # Center position

def servo_position_to_duty_cycle(position):
    """Convert servo position (-1 to 1) to duty cycle percentage (5-10%)"""
    # -1 = 5% duty cycle (1ms pulse), 0 = 7.5% (1.5ms), 1 = 10% (2ms)
    return 7.5 + (position * 2.5)

def duty_cycle_to_position(duty_cycle):
    """Convert duty cycle percentage back to position (-1 to 1)"""
    return (duty_cycle - 7.5) / 2.5

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

    # this method manually moves the servos in the corresponding direction
    def moveLeft(self):
        global currentXServoDuty
        current_pos = duty_cycle_to_position(currentXServoDuty)
        new_pos = max(-1, current_pos - 0.1)
        currentXServoDuty = servo_position_to_duty_cycle(new_pos)
        x_servo_pwm.ChangeDutyCycle(currentXServoDuty)
        print(f"move left - X servo duty: {currentXServoDuty:.2f}%")

    def moveRight(self):
        global currentXServoDuty
        current_pos = duty_cycle_to_position(currentXServoDuty)
        new_pos = min(1, current_pos + 0.1)
        currentXServoDuty = servo_position_to_duty_cycle(new_pos)
        x_servo_pwm.ChangeDutyCycle(currentXServoDuty)
        print(f"move right - X servo duty: {currentXServoDuty:.2f}%")

    def moveUp(self):
        global currentYServoDuty
        current_pos = duty_cycle_to_position(currentYServoDuty)
        new_pos = max(-1, current_pos - 0.1)
        currentYServoDuty = servo_position_to_duty_cycle(new_pos)
        y_servo_pwm.ChangeDutyCycle(currentYServoDuty)
        print(f"move up - Y servo duty: {currentYServoDuty:.2f}%")

    def moveDown(self):
        global currentYServoDuty
        current_pos = duty_cycle_to_position(currentYServoDuty)
        new_pos = min(1, current_pos + 0.1)
        currentYServoDuty = servo_position_to_duty_cycle(new_pos)
        y_servo_pwm.ChangeDutyCycle(currentYServoDuty)
        print(f"move down - Y servo duty: {currentYServoDuty:.2f}%")

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
        global currentXServoDuty, currentYServoDuty
        currentXServoDuty = 7.5  # Center position
        currentYServoDuty = 7.5  # Center position
        x_servo_pwm.ChangeDutyCycle(currentXServoDuty)
        y_servo_pwm.ChangeDutyCycle(currentYServoDuty)
        print("Servos centered")

    # this method starts the GUI loop IMPORTANT: once this is called the program thread will be locked in the loop
    # this is why its important to start the other thread before calling this method
    def startGui(self):
        self.root.mainloop()

# this method is going to be sent to the second thread to run the eye tracking code and send it to the servos over GPIO
def trackEyes():
    global currentXServoDuty, currentYServoDuty # defines the global variables we will use to be global so we can edit them

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

                        error_x = target_x - frame_center_x
                        error_y = target_y - frame_center_y

                        # Calculate servo adjustments
                        servo_adjust_x = error_x * TRACKING_SENSITIVITY_X
                        servo_adjust_y = error_y * TRACKING_SENSITIVITY_Y

                        # Convert current duty cycles to positions for calculation
                        current_x_pos = duty_cycle_to_position(currentXServoDuty)
                        current_y_pos = duty_cycle_to_position(currentYServoDuty)

                        # Apply smoothing and update servo positions
                        new_x_pos = current_x_pos + servo_adjust_x
                        new_y_pos = current_y_pos + servo_adjust_y

                        # Smooth the movement
                        smoothed_x_pos = current_x_pos * SERVO_SMOOTHING + new_x_pos * (1 - SERVO_SMOOTHING)
                        smoothed_y_pos = current_y_pos * SERVO_SMOOTHING + new_y_pos * (1 - SERVO_SMOOTHING)

                        # Clamp to servo limits
                        smoothed_x_pos = max(-1, min(1, smoothed_x_pos))
                        smoothed_y_pos = max(-1, min(1, smoothed_y_pos))

                        # Convert back to duty cycles
                        currentXServoDuty = servo_position_to_duty_cycle(smoothed_x_pos)
                        currentYServoDuty = servo_position_to_duty_cycle(-smoothed_y_pos)  # Invert Y axis

                        # Move servos using hardware PWM
                        x_servo_pwm.ChangeDutyCycle(currentXServoDuty)
                        y_servo_pwm.ChangeDutyCycle(currentYServoDuty)

                        # Add tracking status to frame
                        cv2.putText(frame, "TRACKING", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.putText(frame, f"Servo X: {currentXServoDuty:.2f}%", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        cv2.putText(frame, f"Servo Y: {currentYServoDuty:.2f}%", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        
                        print(f"Tracking - Eyes: ({eye_center_x}, {eye_center_y}) Servo: ({currentXServoDuty:.2f}, {currentYServoDuty:.2f})")

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

# Cleanup function to stop PWM and cleanup GPIO
def cleanup():
    try:
        x_servo_pwm.stop()
        y_servo_pwm.stop()
        GPIO.cleanup()
        print("GPIO cleaned up")
    except:
        pass

# This is the main entry point of the program
if __name__ == "__main__":
    try:
        # Start tracking in a separate thread
        trackingThread = threading.Thread(target=trackEyes, daemon=True)
        trackingThread.start()
        
        # Initialize and Start GUI in main thread
        gui = Gui()
        gui.startGui()
        
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
    finally:
        # Clean up once gui is closed
        cleanup()
        print("Program ended")