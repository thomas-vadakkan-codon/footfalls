class TrackableObject:
	def __init__(self, objectID, centroid):
		self.objectID = objectID # unique ID of the object
		self.centroids = [centroid] # list of centroids of the object
		# to check if the object has been counted or not
		self.counted = False