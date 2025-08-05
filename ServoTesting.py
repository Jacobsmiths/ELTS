from gpiozero import Servo
from threading import Event, Thread
import time 
from gpiozero.pins.pigpio import PiGPIOFactory

quitApplication = False

# these are the GPIO pins for the servos (To do: change these to your actual GPIO pins)
xServoPin = 12

Servo.pin_factory=PiGPIOFactory()
# we will be using gpiozeros servo class to control the servos with PWM
servo = Servo(pin =xServoPin, min_pulse_width=0.0005, max_pulse_width=0.00245)  # min and max pulse widths are found on the servos datasheet
servo.min()


class MyThread(Thread):
    def __init__(self, event):
        Thread.__init__(self)
        self.stopped = event

    def switchPos(self):
        servo.min()
        print("min")
        time.sleep(2)
        servo.mid()
        print("mid")
        time.sleep(2)
        servo.value = 1
        print("max")

    def run(self):
        while not self.stopped.wait(2):
            self.switchPos()


stopFlag = Event()
thread = MyThread(stopFlag)
thread.start()