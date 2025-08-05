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
import GUI

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
        cap.set(cv2.CAP_PROP_FPS, 60)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        print("Camera settings:")
        print(f"  Width:  {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}")
        print(f"  Height: {cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
        print(f"  FPS:    {cap.get(cv2.CAP_PROP_FPS)}")
        
        
        
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

        # Optimize camera settings for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Reduced resolution for better performance
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)  # Set FPS
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer size to avoid lag
        
        # Check if settings were applied
        actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"Camera settings - Width: {actual_width}, Height: {actual_height}, FPS: {actual_fps}")

        # Initialize mediapipe with optimized settings for better performance
        with mp_face_mesh.FaceMesh(
            max_num_faces=1, 
            refine_landmarks=True, 
            min_detection_confidence=0.7,  # Increased for better stability
            min_tracking_confidence=0.5
        ) as face_mesh:

            frame_count = 0
            fps_start_time = time.time()
            
            # this is the second looped thread that will run the eye tracking code indefinitely until stopped 
            while cap.isOpened() and not quitApplication:
                
                # Read current frame
                success, frame = cap.read()
                if not success:
                    print("ERROR - Could not read frame from camera")
                    continue  # Try next frame instead of breaking
                
                # Flip frame horizontally for mirror effect (more intuitive)
                frame = cv2.flip(frame, 1)
                
                # Get frame dimensions
                h, w, _ = frame.shape
                
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
                        
                        # Move servos (uncomment when servos are connected)
                        # xServo.value = currentXServoPos
                        # yServo.value = currentYServoPos
                        
                        # Add tracking status to frame
                        cv2.putText(frame, "TRACKING", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.putText(frame, f"Servo X: {currentXServoPos:.3f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        cv2.putText(frame, f"Servo Y: {currentYServoPos:.3f}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        
                        print(f"Tracking - Eyes: ({eye_center_x}, {eye_center_y}) Error: ({error_x}, {error_y}) Servo: ({currentXServoPos:.3f}, {currentYServoPos:.3f})")

                    # Store last eye position for offset calibration
                    if hasattr(gui, 'lastEyeX'):
                        gui.lastEyeX = eye_center_x
                        gui.lastEyeY = eye_center_y

                else:
                    # No face detected
                    cv2.putText(frame, "NO FACE DETECTED", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # Calculate and display FPS
                frame_count += 1
                if frame_count % 30 == 0:  # Update FPS every 30 frames
                    fps_end_time = time.time()
                    fps = 30 / (fps_end_time - fps_start_time)
                    fps_start_time = fps_end_time
                    print(f"FPS: {fps:.1f}")

                # Display FPS on frame
                cv2.putText(frame, f"FPS: {cap.get(cv2.CAP_PROP_FPS):.1f}", (w-150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

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
    gui = GUI(startCommand=lambda: print("start"), stopCommand=lambda: print("stop"), setXCommand=lambda:print("set x"), setYCommand=lambda:print("set y"),
                centerServosCommand=lambda:print('center servos'), upCommand=lambda:print("up"), downCommand=lambda:print("down"), leftCommand=lambda:print('left'),
                rightCommand=lambda:print("right"))
    gui.start()
    
    # Clean up once gui is closed
    print("Program ended")