# First, make sure you have python3-full installed
sudo apt install python3-full python3-venv

# Create a virtual environment in your project folder
cd ~/Desktop/hand_gesture
python3 -m venv gesture_env

# Activate the virtual environment
source gesture_env/bin/activate

# Now install the packages (you should see (gesture_env) in your prompt)
pip install opencv-python mediapipe numpy


run this to create a environoment
