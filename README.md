# Sign-Verse (ISL Recognition Model)

This project is an Indian Sign Language (ISL) recognition model that translates ISL gestures into English text. It recognizes static signs, including alphabets and numbers, captured by a camera and provides real-time translation through a web interface.

![](https://github.com/CookieDarb/Sign-Verse/blob/main/demo-sign.gif)

## Overview

The project consists of the following components:

1. **Image Collection Program**: A Python script to collect images of ISL gestures for training the model. It captures images, adjusts aspect ratios, and saves them for further processing.

2. **Recognition Script with Flask**: The core of the project, written in Python using OpenCV, Flask, and TensorFlow. It includes the following functionalities:

    - Hand detection using the HandTrackingModule from the cvzone library.
    - Gesture classification using pre-trained TensorFlow and Keras models.
    - Real-time video processing to recognize ISL gestures and display the corresponding English text translation.
    - Web interface using Flask to provide user interaction and translation capabilities.

3. **HTML Templates and CSS**: The project includes HTML templates for the web interface and CSS stylesheets for frontend styling.

## Usage

To use the ISL recognition model, follow these steps:

**Image Collection:**

1. Run the `image_collection.py` script to collect images of ISL gestures.
2. Press 's' to save images. Images will be saved in the specified folder for further processing.

**Training (Optional):**

- Train the recognition model using the collected images and TensorFlow/Keras. This step is optional if pre-trained models are used.

**Run the Recognition Script:**

1. Start the Flask web server by running the `app2.py` script.
2. Access the web interface through a browser to interact with the recognition model.
3. Navigate to different pages for sign-to-text translation, text-to-sign conversion, and other functionalities.
