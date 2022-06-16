from flask import Flask, render_template, request, redirect, url_for
from mylib.centroidtracker import CentroidTracker
from mylib.trackableobject import TrackableObject
from imutils.video import VideoStream
from imutils.video import FPS
from mylib import config
import time, schedule, csv
import argparse, imutils
import time, dlib, cv2, datetime
import numpy as np

app = Flask(__name__)


t0 = time.time()

@app.route('/')
def run():
	try: 
		# construct the argument parse and parse the arguments
		inside = []
		outside = []
		ap = argparse.ArgumentParser()
		ap.add_argument("-p", "--prototxt", required=False,
			help="path to Caffe 'deploy' prototxt file")
		ap.add_argument("-m", "--model", required=True,
			help="path to Caffe pre-trained model")
		ap.add_argument("-i", "--input", type=str,
			help="path to optional input video file")
		ap.add_argument("-o", "--output", type=str,
			help="path to optional output video file")
		ap.add_argument("-c", "--confidence", type=float, default=0.4,
			help="minimum probability to filter weak detections")
		ap.add_argument("-s", "--skip-frames", type=int, default=30,
			help="# of skip frames between detections")
		args = vars(ap.parse_args())

		# initialize the list of class labels MobileNet SSD was trained to detect
		CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
			"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
			"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
			"sofa", "train", "tvmonitor"]

		# load our serialized model from disk
		net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])


		# if a video path was not supplied, grab a reference to the ip camera
		if not args.get("input", False):
			print("[INFO] Starting the live stream..")
			vs = VideoStream(config.url).start()
			# dont minimise the window
			time.sleep(2.0)
			#time.sleep(2.0)

		# otherwise, grab a reference to the video file
		else:
			print("[INFO] Starting the video..")
			vs = cv2.VideoCapture(args["input"])

		# initialize the video writer 
		writer = None

		# initialize the frame dimensions 
		W = None
		H = None

		ct = CentroidTracker(maxDisappeared=40, maxDistance=50) # instantiate the centroid tracker
		trackers = [] # instantiate the list of trackers
		trackableObjects = {} # instantiate the dictionary of trackable objects
		totalFrames = 0 # initialize the total number of frames processed
		totalDown = 0 # initialize the total number of objects that have moved down
		totalUp = 0 # initialize the total number of objects that have moved up
		x = [] 
		empty=[] 
		empty1=[]

		# start the frames per second throughput estimator
		fps = FPS().start()


		# loop over frames from the video stream
		while True:
			# next frame
			frame = vs.read()
			frame = frame[1] if args.get("input", False) else frame

			# end of the video
			if args["input"] is not None and frame is None:
				break

			# resize frame and convert to RGB
			frame = imutils.resize(frame, width = 500)
			rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

			# if the frame dimensions are empty, set them
			if W is None or H is None:
				(H, W) = frame.shape[:2]
			# if we should write video, initialize the writer
			if args["output"] is not None and writer is None:
				fourcc = cv2.VideoWriter_fourcc(*"mp4v")
				writer = cv2.VideoWriter(args["output"], fourcc, 30,
					(W, H), True)
			rects = []

			# when no person is detected, we will use the object detector
			if totalFrames % args["skip_frames"] == 0:
				trackers = []
				# convert fram to blob and obtain detections
				blob = cv2.dnn.blobFromImage(frame, 0.007000, (W, H), 127.5)
				net.setInput(blob)
				detections = net.forward()
				# loop over the detections
				for i in np.arange(0, detections.shape[2]):
					confidence = detections[0, 0, i, 2] #check the confidence of the detection
					if confidence > args["confidence"]: 
						# get the index of the class label from the detections
						idx = int(detections[0, 0, i, 1])
						if CLASSES[idx] != "person": #if its not a person, ignore it
							continue
						# find the co-ordinates of the bounding box for the object
						box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
						(startX, startY, endX, endY) = box.astype("int")
						# construct dlib rectangle object from the co-ordinates and then update the tracker
						tracker = dlib.correlation_tracker()
						rect = dlib.rectangle(startX, startY, endX, endY)
						tracker.start_track(rgb, rect)
						# add the tracker to our list of trackers 
						trackers.append(tracker)

			# if person is detected, we will use the tracker
			else:
				# loop over the trackers
				for tracker in trackers:
					# update the tracker and grab the updated position
					tracker.update(rgb)
					pos = tracker.get_position()
					# unpack the position object
					startX = int(pos.left())
					startY = int(pos.top())
					endX = int(pos.right())
					endY = int(pos.bottom())
					# add the bounding box coordinates to the rectangles list
					rects.append((startX, startY, endX, endY))

			# draw entrance line on the frame
			cv2.line(frame, (0, H // 2), (W, H // 2), (0, 0, 0), 3)
			# draw vertical line on the frame
			#cv2.line(frame, (W // 2, 0), (W // 2, H), (0, 0, 0), 3)
			cv2.putText(frame, "Entrance", (10, H - ((i * 20) + 200)),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
			# update centroid tracker
			objects = ct.update(rects)
			# loop over the tracked objects
			for (objectID, centroid) in objects.items():
				# check to see if objectID is in the list of trackers
				to = trackableObjects.get(objectID, None)
				# if there is no existing trackable object, create one
				if to is None:
					to = TrackableObject(objectID, centroid)
				# but if it exists
				else:
					# check in which direction the object is moving, then
					y = [c[1] for c in to.centroids]
					direction = centroid[1] - np.mean(y)
					to.centroids.append(centroid)

					# check to see if the object has been counted or not
					if not to.counted:
						# when person is going out of the frame, we will
						# when person walks from left of the frame to the right
						if direction < 0 and centroid[1] < H // 2:
							totalUp += 1
							empty.append(totalUp)
							print("Person ", objectID," went out")
							to.counted = True
							outside.append(objectID)

						# when person is coming in the frame, we will
						elif direction > 0 and centroid[1] > H // 2:
							totalDown += 1
							empty1.append(totalDown)
							print("Person ", objectID," went in")
							inside.append(objectID)
							to.counted = True
							
						x = []
						# find total people inside
						x.append(len(empty1)-len(empty))
						f = open("details.txt", "w")
						#write the details to a file
						f.write("Went In: " + str(inside) + "\n")
						f.write("Went Out: " + str(outside) + "\n")
						f.write("Remaining Inside: " + str(len(empty1)-len(empty)) + "\n")
						f.close()
						#make app return runner function
						runner()
						#app.runner()
				trackableObjects[objectID] = to

				# display id and centroid on frame
				text = "{}".format(objectID)
				cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
					cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
				cv2.circle(frame, (centroid[0], centroid[1]), 4, (255, 255, 255), -1)

			# tuples of info we'll be displaying on frame
			info = [
			("Exit", totalUp),
			("Enter", totalDown),
			]

			info2 = [
			("Total people inside", x),
			]

			# Display the output on the frame
			for (i, (k, v)) in enumerate(info):
				text = "{}: {}".format(k, v)
				cv2.putText(frame, text, (10, H - ((i * 20) + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

			for (i, (k, v)) in enumerate(info2):
				text = "{}: {}".format(k, v)
				cv2.putText(frame, text, (265, H - ((i * 20) + 60)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
					
			# check to see if we should write the frame to disk
			if writer is not None:
				writer.write(frame)

			# show the output frame without minimising the window
			cv2.imshow("CodonSoft People Tracker", frame)
			#dont minimise frame
			#cv2.setWindowProperty("CodonSoft People Tracker", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
			key = cv2.waitKey(1) & 0xFF

			# if the `q` key was pressed, break from the loop
			if key == ord("q"):
				break

			#fps updater
			totalFrames += 1
			fps.update()

		# display time and fps information
		fps.stop()
		print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
		print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
		#quit program
		cv2.destroyAllWindows()
		cv2.release(frame)

		# # if we are not using a video file, stop the camera video stream
		# if not args.get("input", False):
		# 	vs.stop()
		#
		# # otherwise, release the video file pointer
		# else:
		# 	vs.release()

		cv2.destroyAllWindows()
		return "Hello"

	except:
		#do nothing
		#return contents of details.txt\
		f = open("details.txt", "r")
		return f.read()

def runner():
	f = open("details.txt", "r")
	return f.read()

if config.Scheduler:
	#schedule whenever u want to run the code
	schedule.every().day.at("09:00").do(run)

	while 1:
		schedule.run_pending()

else:
	app.run()
