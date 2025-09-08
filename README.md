# Driver Drowsiness Detection System

## Overview
This project is a **Driver Drowsiness Detection System** built using Python, OpenCV, and deep learning.  
It monitors the driver's eye state (open or closed) in real time using a webcam feed. If drowsiness is detected (eyes remain closed beyond a threshold), the system plays an alarm sound to alert the driver.

## Features
- Real-time detection of drowsiness using webcam
- Deep learning–based eye state classification
- Audible alarm system (`alarm.wav`) to prevent accidents
- Preloaded dataset of open/closed eye images for training

## Project Structure


Driver Drowsiness System/
│── drowsiness.py # Main program for real-time detection
│── code_1.py # Model training / helper script
│── alarm.wav # Alarm sound file
│── data/
│ └── train/
│ ├── Closed/ # Images of closed eyes
│ └── Open/ # Images of open eyes (if included)


## Requirements
Install the following dependencies before running the project:


pip install opencv-python tensorflow keras numpy playsound

Usage

Clone this repository:

git clone https://github.com/your-username/driver-drowsiness-system.git
cd driver-drowsiness-system


Run the detection script:

python drowsiness.py


The system will access your webcam and monitor your eyes.

If drowsiness is detected, an alarm will sound.

----

Dataset

Training data includes labeled images of open and closed eyes.

The model is trained on these images for classification.

You can expand the dataset with more samples for better accuracy.

------

Future Improvements

Integrate face landmark detection for more accurate eye tracking.

Add yawning detection for enhanced drowsiness detection.

Deploy as a mobile or embedded application (e.g., Raspberry Pi + Camera).

----

License

This project is open-source and available under the MIT License.