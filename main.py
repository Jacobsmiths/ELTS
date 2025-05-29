import tkinter as tk
import threading
import cv2
import mediapipe as mp
from gpiozero import Servo

quitApplication = False
tracking = False

xServoPin = 27
yServoPin = 17

xServo = Servo(xServoPin)
yServo = Servo(yServoPin)

# Calibration offsets and servo positions
xOffset = 0
yOffset = 0
currentXServoPos = 0  # Range from -1 to 1
currentYServoPos = 0  # Range from -1 to 1

# Tracking sensitivity (adjust these values to fine-tune responsiveness)
TRACKING_SENSITIVITY_X = 0.002  # How much to move servo per pixel difference
TRACKING_SENSITIVITY_Y = 0.002
SERVO_SMOOTHING = 0.7  # Smoothing factor (0-1, higher = smoother but slower)

class Gui:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("The Awesome Eye Tracker")

        label = tk.Label(self.root, text="Press Start to begin tracking")
        label.pack(pady=10)

        start_button = tk.Button(self.root, text="Start", command=self.setStartTracking)
        start_button.pack(pady=10)

        stop_button = tk.Button(self.root, text="Stop", command=self.stopTracking)
        stop_button.pack(pady=5)

        exit_button = tk.Button(self.root, text="Exit", command=self.endProgram)
        exit_button.pack(pady=10)

        calibrationFrame = tk.Frame(self.root, padx=10, pady=10)
        calibrationFrame.pack(pady=10)

        arrowFrame = tk.Frame(calibrationFrame, bg="lightgray", padx=10, pady=10)
        arrowFrame.pack(side=tk.RIGHT, pady=10)

        tk.Button(arrowFrame, text="left", command=self.moveLeft).grid(column=0, row=1)
        tk.Button(arrowFrame, text="right", command=self.moveRight).grid(column=2, row=1)
        tk.Button(arrowFrame, text="up", command=self.moveUp).grid(column=1, row=0)
        tk.Button(arrowFrame, text="down", command=self.moveDown).grid(column=1, row=2)

        setFrame = tk.Frame(calibrationFrame, padx=10)
        setFrame.pack(side=tk.LEFT)

        tk.Button(setFrame, text="Set Offset For X", command=self.setMaxX).pack(pady=5)
        tk.Button(setFrame, text="Set Offset For Y", command=self.setMaxY).pack(pady=5)
        tk.Button(setFrame, text="Center Servos", command=self.centerServos).pack(pady=5)

    def setStartTracking(self):
        global tracking
        tracking = True
        print("start tracking")

    def stopTracking(self):
        global tracking
        tracking = False
        print("stop tracking")

    def endProgram(self):
        global quitApplication
        quitApplication = True
        print("ending program")
        self.root.quit() 

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

    def centerServos(self):
        global currentXServoPos, currentYServoPos
        currentXServoPos = 0
        currentYServoPos = 0
        xServo.mid()
        yServo.mid()
        print("Servos centered")

    def startGui(self):
        self.root.mainloop()

def trackEyes():
    global currentXServoPos, currentYServoPos, tracking, quitApplication
    
    mp_face_mesh = mp.solutions.face_mesh
    LEFT_IRIS = [474, 475, 476, 477]
    RIGHT_IRIS = [469, 470, 471, 472]

    try: 
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    except Exception as e:
        print(f"Capture is not found: {e}")
        return

    # Center the servos initially
    xServo.mid()
    yServo.mid()
    
    with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=.5) as face_mesh:
        while cap.isOpened() and not quitApplication:
            success, frame = cap.read()
            if not success:
                print("ERROR - Could not read capture")
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            # Calculate center of frame for reference
            frame_center_x = w // 2
            frame_center_y = h // 2

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            if results.multi_face_landmarks:
                mesh = results.multi_face_landmarks[0].landmark
                leftCords = [(int(mesh[p].x * w), int(mesh[p].y * h)) for p in LEFT_IRIS]
                rightCords = [(int(mesh[p].x * w), int(mesh[p].y * h)) for p in RIGHT_IRIS]

                left_center = tuple(map(lambda x: sum(x) // len(x), zip(*leftCords)))
                right_center = tuple(map(lambda x: sum(x) // len(x), zip(*rightCords)))

                # Use the average of both eyes for more stable tracking
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
                cv2.line(frame, (frame_center_x - 20, frame_center_y), (frame_center_x + 20, frame_center_y), (255, 255, 255), 1)
                cv2.line(frame, (frame_center_x, frame_center_y - 20), (frame_center_x, frame_center_y + 20), (255, 255, 255), 1)

                if tracking:
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
                    xServo.value = currentXServoPos
                    yServo.value = currentYServoPos
                    
                    print(f"Tracking - Eyes: ({eye_center_x}, {eye_center_y}) Error: ({error_x}, {error_y}) Servo: ({currentXServoPos:.3f}, {currentYServoPos:.3f})")

                # Store last eye position for offset calibration
                gui.lastEyeX = eye_center_x
                gui.lastEyeY = eye_center_y

            cv2.imshow("Eye Tracker", frame)
            
            # Small delay to prevent excessive CPU usage
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Start tracking in a separate thread
    trackingThread = threading.Thread(target=trackEyes, daemon=True)
    trackingThread.start()
    
    # Start GUI in main thread
    gui = Gui()
    gui.startGui()
    
    # Clean up
    quitApplication = True
    trackingThread.join(timeout=1.0)