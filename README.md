This project is a real-time facial emotion recognition system that uses a webcam to detect a user’s face and classify their emotional expression. The system processes live video, identifies the primary face, and displays the predicted emotion along 
with a confidence score.

It demonstrates a complete pipeline of:
Webcam --> Face Detection --> Emotion Classification --> Live Output

Features: 
* Real-time face detection using OpenCV
* Emotion classification using a pretrained CNN model
* Stable predictions using smoothing techniques
* Confidence score display
* FPS (frames per second) counter
* Logging of detected emotions

Requirements:

Make sure to install both of these packages in the terminal:
pip install opencv-python
pip install tensorflow keras numpy

Of course, you'll need a webcam as well

