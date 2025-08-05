# these are the imports we will use in this program
import tkinter as tk
import threading
import cv2
import mediapipe as mp
from gpiozero import Servo
import csv
import time
import GUI

# defines global variables we will use
# these are flags to control the application state
quitApplication = False
tracking = False

# these are the GPIO pins for the servos (To do: change these to your actual GPIO pins)
xServoPin = 27
yServoPin = 17

# we will be using gpiozeros servo class to control the servos with PWM
# xServo = Servo(xServoPin, min_pulse_width=0.0005, max_pulse_width=0.0025)  # min and max pulse widths are found on the servos datasheet
# yServo = Servo(yServoPin, min_pulse_width=0.0005, max_pulse_width=0.0025) 

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
    
    gui = GUI(startCommand=lambda: print("start"), stopCommand=lambda: print("stop"), setXCommand=lambda:print("set x"), setYCommand=lambda:print("set y"),
                centerServosCommand=lambda:print('center servos'), upCommand=lambda:print("up"), downCommand=lambda:print("down"), leftCommand=lambda:print('left'),
                rightCommand=lambda:print("right"))
    gui.start()
    
    # Clean up once gui is closed
    print("Program ended")