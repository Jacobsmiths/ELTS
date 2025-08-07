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
        
        self.xPid = PIDController(self.X_PID_KP, self.X_PID_KI, self.X_PID_KD, self.FRAME_CENTER_X)
        self.yPid = PIDController(self.Y_PID_KP, self.Y_PID_KI, self.Y_PID_KD, self.FRAME_CENTER_Y)

        self.currTime = time.perf_counter() # TODO might want to move this into the actual phase of tracking instead of having it count up not being

