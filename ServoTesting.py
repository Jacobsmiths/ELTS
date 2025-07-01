import tkinter as tk
from gpiozero import Servo

quitApplication = False

# these are the GPIO pins for the servos (To do: change these to your actual GPIO pins)
xServoPin = 12

# we will be using gpiozeros servo class to control the servos with PWM
servo = Servo(xServoPin, min_pulse_width=0.0005, max_pulse_width=0.0025)  # min and max pulse widths are found on the servos datasheet
servo.min()

pos = IntVar(servo.val)

root = tk.Tk()
root.title("Tester") 

def setServoUp():
    servo.value += 1
    pos+=1

def setServoDown():
    servo.value-=1
    pos-=1


up_button = tk.Button(root, text="Up", command=setServoUp)
up_button.pack(pady=10)

down_button = tk.Button(root, text="Down", command=setServoDown)
down_button.pack(pady=10)

servo_pos = tk.Label(root, text=pos)

