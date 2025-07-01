import timer
from gpiozero import Servo
from gpiozero.pins.pigpio import PiGPIOFactory

quitApplication = False

# these are the GPIO pins for the servos (To do: change these to your actual GPIO pins)
xServoPin = 12

Servo.pin_factory=PiGPIOFactory('127.0.0.1')
# we will be using gpiozeros servo class to control the servos with PWM
servo = Servo(pin =xServoPin, min_pulse_width=0.0005, max_pulse_width=0.0025)  # min and max pulse widths are found on the servos datasheet
servo.min()

positions = ["min", "mid" , "max"]
current = 0

class MyThread(Thread):
    def __init__(self, event):
        Thread.__init__(self)
        self.stopped = event

    def switchPos(self):
        current = (current+1)%3
        if(positions[current]=="min"):
            servo.min()
        elif(posiitons[current]=="mid"):
            servo.mid()
        else:
            servo.max()

    def run(self):
        while not self.stopped.wait(3):
            self.switchPos()


stopFlag = Event()
thread = MyThread(stopFlag)
thread.start()