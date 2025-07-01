from gpiozero import Servo
from threading import Event, Thread
import time 
from gpiozero.pins.pigpio import PiGPIOFactory

quitApplication = False

# these are the GPIO pins for the servos (To do: change these to your actual GPIO pins)
xServoPin = 12

Servo.pin_factory=PiGPIOFactory()
# we will be using gpiozeros servo class to control the servos with PWM
servo = Servo(pin =xServoPin, min_pulse_width=0.0005, max_pulse_width=0.0027, frame_width=0.02)  # min and max pulse widths are found on the servos datasheet
servo.min()

positions = ["min", "mid" , "max"]
current = 0

class MyThread(Thread):
    def __init__(self, event):
        Thread.__init__(self)
        self.stopped = event

    def switchPos(self):
        servo.min()
        print("min")
        time.sleep(5)
        servo.mid()
        print("mid")
        time.sleep(5)
        servo.value = 1
        print("max")

    def run(self):
        while not self.stopped.wait(3):
            self.switchPos()


stopFlag = Event()
thread = MyThread(stopFlag)
thread.start()