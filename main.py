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

class PIDController:
    def __init__(self, Kp, Ki, Kd, setpoint):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        self.previous_error = 0
        self.integral = 0
        self.last_time = None
        
    def compute(self, process_variable, dt):
        # Calculate error
        error = self.setpoint - process_variable
        
        # Proportional term
        P_out = self.Kp * error
        
        # Integral term
        self.integral += error * dt
        I_out = self.Ki * self.integral
        
        # Derivative term
        derivative = (error - self.previous_error) / dt if dt > 0 else 0
        D_out = self.Kd * derivative
        
        # Compute total output
        output = P_out + I_out + D_out
        
        # Update previous error
        self.previous_error = error
        
        return output
    
    def reset(self):
        """Reset PID controller state"""
        self.previous_error = 0
        self.integral = 0
        self.last_time = None
    
    def set_setpoint(self, setpoint):
        """Update the setpoint"""
        self.setpoint = setpoint

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

# PID controller parameters (tuned for smooth eye tracking)
# Start with these conservative values and tune from here
X_PID_KP = 18   # Proportional gain
X_PID_KI = 8 # Integral gain (small to prevent windup)
X_PID_KD = 3  # Derivative gain

Y_PID_KP = 9   # Proportional gain
Y_PID_KI = 1  # Integral gain (small to prevent windup)
Y_PID_KD = 0  # Derivative gain

# Servo limits and constraints
SERVO_MIN_ANGLE = 0
SERVO_MAX_ANGLE = 180
SERVO_CENTER_ANGLE = 90

# Maximum change per update (degrees) - prevents jerky movements
MAX_SERVO_CHANGE = 15  # Smaller for smoother normalized control

# Initialize I2C bus and PCA9685
i2c = busio.I2C(board.SCL, board.SDA)
pca = PCA9685(i2c)
pca.frequency = SERVO_FREQUENCY

# Create servo objects
xServo = servo.Servo(pca.channels[X_SERVO_CHANNEL], min_pulse=SERVO_MIN_PULSE, max_pulse=SERVO_MAX_PULSE)
yServo = servo.Servo(pca.channels[Y_SERVO_CHANNEL], min_pulse=SERVO_MIN_PULSE, max_pulse=SERVO_MAX_PULSE)

# Calibration offsets
xOffset = 0
yOffset = 0

# Servo positions in degrees (typically 0-180 degrees)
currentXServoPos = SERVO_CENTER_ANGLE  # Center position
currentYServoPos = SERVO_CENTER_ANGLE  # Center position

# Frame dimensions for setpoint calculation
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FRAME_CENTER_X = FRAME_WIDTH // 2
FRAME_CENTER_Y = FRAME_HEIGHT // 2

# Initialize PID controllers
# Setpoint is 0 (center), we want to drive the normalized error to zero
x_pid = PIDController(X_PID_KP, X_PID_KI, X_PID_KD, 0)  # Setpoint is 0 (centered)
y_pid = PIDController(Y_PID_KP, Y_PID_KI, Y_PID_KD, 0)  # Setpoint is 0 (centered)

# Timing variables for PID
last_update_time = time.time()

def setServoAngle(servo_obj, angle):
    """Set servo to specific angle (0-180 degrees)"""
    try:
        # Clamp angle to valid range
        angle = max(SERVO_MIN_ANGLE, min(SERVO_MAX_ANGLE, angle))
        servo_obj.angle = angle
    except Exception as e:
        print(f"Error setting servo angle: {e}")

def centerServos():
    """Center both servos to 90 degrees and reset PID controllers"""
    global currentXServoPos, currentYServoPos, last_update_time
    
    currentXServoPos = SERVO_CENTER_ANGLE
    currentYServoPos = SERVO_CENTER_ANGLE
    setServoAngle(xServo, SERVO_CENTER_ANGLE)
    setServoAngle(yServo, SERVO_CENTER_ANGLE)
    
    # Reset PID controllers
    x_pid.reset()
    y_pid.reset()
    last_update_time = time.time()
    
    print("Servos centered at 90 degrees and PIDs reset")

def updateServoWithPID(eye_center_x, eye_center_y):
    """Update servo positions using PID control with normalized values"""
    global currentXServoPos, currentYServoPos, last_update_time
    
    current_time = time.time()
    dt = current_time - last_update_time
    
    # Minimum time delta to prevent division by zero
    if dt < 0.001:
        return 0, 0
    
    # Apply offsets to eye position
    target_x = eye_center_x + xOffset
    target_y = eye_center_y + yOffset
    
    # Calculate normalized error (-1 to 1)
    # Negative error means eyes are left/up of center, positive means right/down
    error_x = (target_x - FRAME_CENTER_X) / FRAME_CENTER_X  # Normalize to -1 to 1
    error_y = (target_y - FRAME_CENTER_Y) / FRAME_CENTER_Y  # Normalize to -1 to 1
    
    # Compute PID outputs (normalized values)
    # PID setpoint is 0 (center), process variable is the normalized error
    x_output = x_pid.compute(error_x, dt)
    y_output = y_pid.compute(error_y, dt)
    
    # Apply PID output directly to servo position
    # We want to move servo in opposite direction of error to center the target
    servo_adjust_x = -x_output  # Negative because we want to counteract the error
    servo_adjust_y = -y_output
    
    # Limit maximum change per update to prevent jerky movements
    if abs(servo_adjust_x) > MAX_SERVO_CHANGE:
        servo_adjust_x = MAX_SERVO_CHANGE if servo_adjust_x > 0 else -MAX_SERVO_CHANGE
    if abs(servo_adjust_y) > MAX_SERVO_CHANGE:
        servo_adjust_y = MAX_SERVO_CHANGE if servo_adjust_y > 0 else -MAX_SERVO_CHANGE
    
    # Apply changes to current positions
    currentXServoPos += servo_adjust_x
    currentYServoPos += servo_adjust_y
    
    # Clamp to servo limits
    currentXServoPos = max(SERVO_MIN_ANGLE, min(SERVO_MAX_ANGLE, currentXServoPos))
    currentYServoPos = max(SERVO_MIN_ANGLE, min(SERVO_MAX_ANGLE, currentYServoPos))
    
    # Move servos
    setServoAngle(xServo, currentXServoPos)
    setServoAngle(yServo, currentYServoPos)
    
    # Update timing
    last_update_time = current_time
    
    return x_output, y_output

def trackEyes():
    global currentXServoPos, currentYServoPos, last_update_time
    
    startTime = time.time()
    
    with open('data.csv', 'w', newline='') as csvfile:
        fieldnames = ['time', 'x', 'y', 'servo_x', 'servo_y', 'pid_x_output', 'pid_y_output']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # MediaPipe setup
        mp_face_mesh = mp.solutions.face_mesh
        LEFT_IRIS = [474, 475, 476, 477]
        RIGHT_IRIS = [469, 470, 471, 472]
        
        # Camera setup
        cap = None
        for camera_index in [0, 1, 2]:
            try:
                print(f"Trying camera index {camera_index}...")
                cap = cv2.VideoCapture(camera_index)
                
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
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        print("Camera settings:")
        print(f"  Width:  {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}")
        print(f"  Height: {cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
        print(f"  FPS:    {cap.get(cv2.CAP_PROP_FPS)}")
        
        # Initialize servos to center position
        centerServos()
        
        # Initialize mediapipe
        with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.1
        ) as face_mesh:
            
            while cap.isOpened() and not quitApplication:
                success, frame = cap.read()
                if not success:
                    print("ERROR - Could not read frame from camera")
                    continue
                
                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Process frame
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                results = face_mesh.process(rgb)
                rgb.flags.writeable = True
                
                pid_x_output = pid_y_output = 0
                
                if results.multi_face_landmarks:
                    mesh = results.multi_face_landmarks[0].landmark
                    
                    # Get iris coordinates
                    leftCords = [(int(mesh[p].x * FRAME_WIDTH), int(mesh[p].y * FRAME_HEIGHT)) for p in LEFT_IRIS]
                    rightCords = [(int(mesh[p].x * FRAME_WIDTH), int(mesh[p].y * FRAME_HEIGHT)) for p in RIGHT_IRIS]
                    
                    # Calculate eye centers
                    left_center = tuple(map(lambda x: sum(x) // len(x), zip(*leftCords)))
                    right_center = tuple(map(lambda x: sum(x) // len(x), zip(*rightCords)))
                    
                    # Average of both eyes for stable tracking
                    eye_center_x = (left_center[0] + right_center[0]) // 2
                    eye_center_y = (left_center[1] + right_center[1]) // 2
                    
                    # Apply offsets
                    target_x = eye_center_x 
                    target_y = eye_center_y 
                    
                    # Draw tracking indicators
                    cv2.circle(frame, left_center, 3, (0, 255, 0), -1)
                    cv2.circle(frame, right_center, 3, (0, 255, 0), -1)
                    cv2.circle(frame, (eye_center_x, eye_center_y), 5, (255, 0, 0), 2)
                    cv2.circle(frame, (target_x, target_y), 7, (0, 0, 255), 2)
                    
                    # Draw crosshairs at frame center
                    cv2.line(frame, (FRAME_CENTER_X - 20, FRAME_CENTER_Y), (FRAME_CENTER_X + 20, FRAME_CENTER_Y), (255, 255, 255), 2)
                    cv2.line(frame, (FRAME_CENTER_X, FRAME_CENTER_Y - 20), (FRAME_CENTER_X, FRAME_CENTER_Y + 20), (255, 255, 255), 2)
                    
                    if tracking:
                        # Update servos using PID control
                        pid_x_output, pid_y_output = updateServoWithPID(eye_center_x, eye_center_y)
                        
                        # Log data to CSV
                        writer.writerow({
                            "time": time.time() - startTime,
                            "x": eye_center_x,
                            "y": eye_center_y,
                            "servo_x": currentXServoPos,
                            "servo_y": currentYServoPos,
                            "pid_x_output": pid_x_output,
                            "pid_y_output": pid_y_output
                        })
                        csvfile.flush()
                        
                        # Display tracking status
                        cv2.putText(frame, "PID TRACKING", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.putText(frame, f"Servo X: {currentXServoPos:.1f}°", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        cv2.putText(frame, f"Servo Y: {currentYServoPos:.1f}°", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        cv2.putText(frame, f"PID X: {pid_x_output:.2f}", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        cv2.putText(frame, f"PID Y: {pid_y_output:.2f}", (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        
                        # Calculate and display normalized error
                        error_x_norm = (target_x - FRAME_CENTER_X) / FRAME_CENTER_X
                        error_y_norm = (target_y - FRAME_CENTER_Y) / FRAME_CENTER_Y
                        cv2.putText(frame, f"Error X: {error_x_norm:.3f}", (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        cv2.putText(frame, f"Error Y: {error_y_norm:.3f}", (10, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        
                        print(f"PID Tracking - Eyes: ({eye_center_x}, {eye_center_y}) Servo: ({currentXServoPos:.1f}°, {currentYServoPos:.1f}°) Error: ({error_x_norm:.3f}, {error_y_norm:.3f})")
                    
                    # Store last eye position for GUI
                    if hasattr(gui, 'lastEyeX'):
                        gui.lastEyeX = eye_center_x
                        gui.lastEyeY = eye_center_y
                
                else:
                    # No face detected - reset PID controllers to prevent integral windup
                    if tracking:
                        x_pid.reset()
                        y_pid.reset()
                    cv2.putText(frame, "NO FACE DETECTED", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # Display the frame
                cv2.imshow("PID Eye Tracker", frame)
                
                # Check for 'q' key press to quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    
    # Cleanup
    if cap:
        cap.release()
    cv2.destroyAllWindows()
    print("Camera and windows cleaned up")

# Additional utility functions for PID tuning
def tunePID(axis, kp, ki, kd):
    """Adjust PID parameters during runtime"""
    global x_pid, y_pid
    
    if axis.lower() == 'x':
        x_pid.Kp = kp
        x_pid.Ki = ki
        x_pid.Kd = kd
        x_pid.reset()  # Reset to apply new parameters
        print(f"X-axis PID tuned: Kp={kp}, Ki={ki}, Kd={kd}")
    elif axis.lower() == 'y':
        y_pid.Kp = kp
        y_pid.Ki = ki
        y_pid.Kd = kd
        y_pid.reset()
        print(f"Y-axis PID tuned: Kp={kp}, Ki={ki}, Kd={kd}")
    else:
        print("Invalid axis. Use 'x' or 'y'")

def resetPIDControllers():
    """Reset both PID controllers"""
    global x_pid, y_pid
    x_pid.reset()
    y_pid.reset()
    print("PID controllers reset")


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



# Main entry point
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