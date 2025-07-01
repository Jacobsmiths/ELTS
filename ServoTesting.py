import tkinter as tk
from gpiozero import Servo
from gpiozero.pins.pigpio import PiGPIOFactory

quitApplication = False

# these are the GPIO pins for the servos (To do: change these to your actual GPIO pins)
xServoPin = 12

Servo.pin_factory=PiGPIOFactory('127.0.0.1')
# we will be using gpiozeros servo class to control the servos with PWM
servo = Servo(pin =xServoPin, min_pulse_width=0.0005, max_pulse_width=0.0025)  # min and max pulse widths are found on the servos datasheet
servo.min()

# pos = tk.IntVar(value=servo.value)


def setServoUp():
    servo.max()
    # pos+=0.1

def setServoDown():
    servo.min()
       # pos-=0.01


root = tk.Tk()
root.title("Tester") 
up_button = tk.Button(root, text="Up", command=setServoUp)
up_button.pack(pady=10)

down_button = tk.Button(root, text="Down", command=setServoDown)
down_button.pack(pady=10)

# servo_pos = tk.Label(root, text=pos)


root.mainloop()
