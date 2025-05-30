Welcome to the ELTS system code!
Getting started:
To install the code you can either clone git repo or you can just download the code and unzip the file.
Next you will need to install the libraries, I would advise making an environment first and installing libraries there.
The reason for this is to keep an organized version of your libraries and your python, otherwise it can lead to a lot of out dated libraries and inability to run code.
To make an environment in Python: 
  python3 -m venv env
  
this will make a folder called env that you can write the libraries to.
to activate the venv it depends on your system but its either:
  (windows): ./env/Scripts/activate
  (unix): source ./env/bin/activate

then install the libraries:
  pip install -r requirements.txt

And then your good to go. Run with:
  python main.py

Note: Running the servo code on anything that doesn't have GPIO pins will cause an error I think so if you want to just run the eye tracking and GUI 
then you need to comment out the line that instatiates the servos, called xServo and yServo = ... then also comment out everything that uses those servos
But then everything should run and work normally.
