# PeopleCounter in Real Time
People Counting in Real-Time using live video stream/IP camera in OpenCV.

> This is an improvement/modification to https://www.pyimagesearch.com/2018/08/13/opencv-people-counter/ which i have developed for CodonSoft 

- The major goal is to leverage the project as a ready-to-scale business model.
- Use case: real-time counting of people in stores, buildings, shopping malls, and other locations.
- Improving performance by automating features and optimising the real-time stream.
- Also serves as a countermeasure to COVID-19 by allowing for footfall analysis.

--- 

## Table of Contents
* [Introduction](#introduction)
* [Running Inference](#running-inference)
* [Features](#features)
* [References](#references)
* [Limitations](#limitations)

## Introduction
**SSD detector:**
- We're employing a MobileNet architecture with an SSD (Single Shot Detector). In most cases, only one shot is required to detect everything is present in an image. That is, one for generating region suggestions and another for determining what each proposal's purpose is.
- SSD is quite fast when compared to other 2 shot detectors like R-CNN.
- MobileNet is a DNN designed to run on low-resource devices, as the name suggests. Mobile phones, IP cameras, scanners, and other similar devices are examples.
- As a result, a MobileNet seasoned SSD should ideally result in a faster, more efficient object detection.
---
**Centroid tracker:**
- The Centroid Tracker is one of the most trustworthy trackers available.
- The centroid tracker, to put it simply, calculates the centroid of the bounding boxes.
- The bounding boxes, in other words, are the (x, y) coordinates of the objects in a picture.
- The tracker computes the centroid (centre) of the box once our SSD obtains the co-ordinates. To put it another way, the object's core.
- Then, for tracking purposes over the sequence of frames, each detected object is given a unique ID.

## Running Inference
- Install all the required Python dependencies:
```
pip install -r requirements.txt
```
- To run inference on a test video file:
- Setup path of your test video in the main.py file
```
# Enter the path of your video at the end of the line. For example:
os.system("python Run.py --prototxt mobilenet_ssd/MobileNetSSD_deploy.prototxt --model mobilenet_ssd/MobileNetSSD_deploy.caffemodel --input videos/example_01.mp4")
```
- Then run with the command: 
```
python main.py
```
- To run inference on an IP camera:
- Setup your camera url in 'mylib/config.py':

```
# Enter the ip camera url (e.g., url = 'rtsp://codonsoft:rohanchowdary@192.168.0.131:554/stream1')
url = ''
```
- Then run with the command:
```
python manage.py
```
> Set url = 0 for webcam.

## Features

***1. Scheduler:***
- Automatic scheduler to start the software. Configure to run at every second, minute, day, or Monday to Friday.
- This is extremely useful in a business scenario, for instance, you can run it only at your desired time (9-5?).
- Variables and memory would be reset == less load on your machine.

```
##Runs at every day (9:00 am). You can change it.
schedule.every().day.at("9:00").do(run)
```

***2. Simple log:***
- Logs all data during every run.
- Useful for footfall analysis.
- All the details of every person going in, coming out, and remaining inside can be seen in the details.txt file

## References
- SSD paper: https://arxiv.org/abs/1512.02325
- MobileNet paper: https://arxiv.org/abs/1704.04861
- Centroid tracker: https://www.pyimagesearch.com/2018/07/23/simple-object-tracking-with-opencv/
- https://towardsdatascience.com/review-ssd-single-shot-detector-object-detection-851a94607d11
- https://pypi.org/project/schedule/

## Limitations
- Train the SSD on human data only with a top-down view for good accuracy.
- Experiment with other detectors and benchmark the results on computationally less expensive embedded hardware. 
- Evaluate the performance on multiple IP cameras.

<p>&nbsp;</p>

---

