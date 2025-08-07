import tkinter as tk

class GUI:
    def __init__(self, startCommand, stopCommand, centerServosCommand, resetOffsetsCommand, upCommand, downCommand, leftCommand, rightCommand):
        self.root = tk.Tk()
        self.root.title("ELTS")
        self.root.geometry("500x400")

        title = tk.Label(self.root, text="ELTS Control Board", font=("bold", 20))
        title.pack(pady=10, padx=10)

        exitFrame = tk.Frame(self.root)
        exitFrame.pack(pady=5)
        exitLabel = tk.Label(exitFrame, text="Press to exit Program")
        exitLabel.pack(side="left", padx=5)
        exitButton = tk.Button(exitFrame, text="Exit", command=lambda: self.root.quit())
        exitButton.pack(side="left", padx=5)

        startFrame = tk.Frame(self.root)
        startFrame.pack(pady=5)
        startLabel = tk.Label(startFrame, text="Press to begin tracking") 
        startLabel.pack(side="left", padx=5)
        startButton = tk.Button(startFrame, text="Start", command=startCommand)
        startButton.pack(side="left", padx=5)

        stopFrame = tk.Frame(self.root)
        stopFrame.pack(pady=5)
        stopLabel = tk.Label(stopFrame, text="Press to stop tracking")
        stopLabel.pack(side="left", padx=5)
        stopButton = tk.Button(stopFrame, text="Stop", command=stopCommand)
        stopButton.pack(side="left", padx=5)

        calibrationFrame = tk.Frame(self.root)
        calibrationFrame.pack(pady=10)

        arrowFrame = tk.Frame(calibrationFrame, bg="lightgray", padx=10)
        arrowFrame.pack(side=tk.RIGHT)

        tk.Button(arrowFrame, text="<", command=leftCommand).grid(column=0, row=1)
        tk.Button(arrowFrame, text=">", command=rightCommand).grid(column=2, row=1)
        tk.Button(arrowFrame, text="^", command=upCommand).grid(column=1, row=0)
        tk.Button(arrowFrame, text="v", command=downCommand).grid(column=1, row=2)

        setFrame = tk.Frame(calibrationFrame, padx=10)
        setFrame.pack(side=tk.LEFT)

        tk.Button(setFrame, text="Clear offsets", command=resetOffsetsCommand).pack(pady=5)
        tk.Button(setFrame, text="Center Servos", command=centerServosCommand).pack(pady=5)

    def start(self):
        self.root.mainloop()

# this is purely for testign the gui
if __name__ == "__main__":
    gui = GUI(startCommand=lambda: print("start"), stopCommand=lambda: print("stop"), resetOffsetsCommand=lambda:print("reset offsets"),
                centerServosCommand=lambda:print('center servos'), upCommand=lambda:print("up"), downCommand=lambda:print("down"),
                leftCommand=lambda:print('left'), rightCommand=lambda:print("right"))
    gui.start()