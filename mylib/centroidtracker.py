# import the necessary packages
from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np

class CentroidTracker:
	def __init__(self, maxDisappeared=50, maxDistance=50):
		self.nextObjectID = 1 # next object ID to be assigned
		self.objects = OrderedDict() # object ID -> centroid
		self.disappeared = OrderedDict() # object ID -> number of consecutive frames it has been marked as "disappeared"

		# max consecutive frames to ignore an object
		self.maxDisappeared = maxDisappeared

		#max distance b/w centroids to be considered the same object
		self.maxDistance = maxDistance

	def register(self, centroid):
		# set objectID to next available object ID
		self.objects[self.nextObjectID] = centroid
		self.disappeared[self.nextObjectID] = 1
		self.nextObjectID += 1

	def deregister(self, objectID):
		# deregister an object ID
		del self.objects[objectID]
		del self.disappeared[objectID]

	def update(self, rects):
		# check if list of bounding box rectangles is empty
		if len(rects) == 0:
			# loop existing tracked objects and mark them as disappeared
			for objectID in list(self.disappeared.keys()):
				self.disappeared[objectID] += 1

				#if max consecutive frames reached, deregister object
				if self.disappeared[objectID] > self.maxDisappeared:
					self.deregister(objectID)

			# return when no updates needed
			return self.objects

		# initialize an array of input centroids for the current frame
		inputCentroids = np.zeros((len(rects), 2), dtype="int")

		# loop over the bounding box rectangles
		for (i, (startX, startY, endX, endY)) in enumerate(rects):
			# use the bounding box coordinates to derive the centroid
			cX = int((startX + endX) / 2.0)
			cY = int((startY + endY) / 2.0)
			inputCentroids[i] = (cX, cY)

		# register each input centroid if it is not currently being tracked
		if len(self.objects) == 0:
			for i in range(0, len(inputCentroids)):
				self.register(inputCentroids[i])

		# or else match input centroids to tracked objects
		else:
			# grab the set of object IDs and corresponding centroids
			objectIDs = list(self.objects.keys())
			objectCentroids = list(self.objects.values())

			# find distamce between each input centroid and the object centroids
			D = dist.cdist(np.array(objectCentroids), inputCentroids)

			# sort rows and columns to find the nearest object centroid for each input centroid
			rows = D.min(axis=1).argsort()
			cols = D.argmin(axis=1)[rows]

			# check rows and columns to see if object must be registered, deregistered, or updated
			usedRows = set()
			usedCols = set()

			# loop over row, column index pairs
			for (row, col) in zip(rows, cols):
				# if we have already examined either the row or column value before, ignore it
				if row in usedRows or col in usedCols:
					continue

				# when distance is less than maxDistance, it is a new object
				if D[row, col] > self.maxDistance:
					continue

				# or else update the tracked object centroid
				objectID = objectIDs[row]
				self.objects[objectID] = inputCentroids[col]
				self.disappeared[objectID] = 0

				# we examined this object
				usedRows.add(row)
				usedCols.add(col)

			# compute both row and column not yet examined
			unusedRows = set(range(0, D.shape[0])).difference(usedRows)
			unusedCols = set(range(0, D.shape[1])).difference(usedCols)

			# when num_disappeared is greater than maxDisappeared, deregister the object
			if D.shape[0] >= D.shape[1]:
				# loop over the unused row indexes
				for row in unusedRows:
					# increment disappeared count
					objectID = objectIDs[row]
					self.disappeared[objectID] += 1

					# check to see if object can be deregistered
					if self.disappeared[objectID] > self.maxDisappeared:
						self.deregister(objectID)

			# or else register the object
			else:
				for col in unusedCols:
					self.register(inputCentroids[col])

		# return the set of trackable objects
		return self.objects