# USAGE
# python yolo_video.py --input videos/airport.mp4 --output output/airport_output.avi --yolo yolo-coco

# import the necessary packages
import numpy as np
import argparse
import imutils
import time
import cv2
import os

def initializeTracker(tracker_type,minor_ver):
	if int(minor_ver) < 3:
		tracker = cv2.Tracker_create(tracker_type)
	else:
		if tracker_type == 'BOOSTING':
			tracker = cv2.TrackerBoosting_create()
		if tracker_type == 'MIL':
			tracker = cv2.TrackerMIL_create()
		if tracker_type == 'KCF':
			tracker = cv2.TrackerKCF_create()
		if tracker_type == 'TLD':
			tracker = cv2.TrackerTLD_create()
		if tracker_type == 'MEDIANFLOW':
			tracker = cv2.TrackerMedianFlow_create()
		if tracker_type == 'GOTURN':
			tracker = cv2.TrackerGOTURN_create()
		if tracker_type == 'MOSSE':
			tracker = cv2.TrackerMOSSE_create()
		if tracker_type == "CSRT":
			tracker = cv2.TrackerCSRT_create()
	return tracker

def get_IOU(bbox1, bbox2):
	if bbox1[0] > bbox2[0]+bbox2[2] or bbox2[0] > bbox1[0]+bbox1[2]:
		return 0
	if bbox1[1] > bbox2[1]+bbox2[3] or bbox2[1] > bbox1[1]+bbox1[3]:
		return 0
	minx = min(bbox1[0]+bbox1[2],bbox2[0]+bbox2[2])
	miny = min(bbox1[1]+bbox1[3],bbox2[1]+bbox2[3])
	maxx = max(bbox1[0],bbox2[0])
	maxy = max(bbox1[1],bbox2[1])
	intersection = (maxx-minx)*(maxy-miny)
	return intersection/(bbox1[2]*bbox1[3] + bbox2[2]*bbox2[3] - intersection)

def get_IO1(bbox1, bbox2):
	if bbox1[0] > bbox2[0]+bbox2[2] or bbox2[0] > bbox1[0]+bbox1[2]:
		return 0
	if bbox1[1] > bbox2[1]+bbox2[3] or bbox2[1] > bbox1[1]+bbox1[3]:
		return 0
	minx = min(bbox1[0]+bbox1[2],bbox2[0]+bbox2[2])
	miny = min(bbox1[1]+bbox1[3],bbox2[1]+bbox2[3])
	maxx = max(bbox1[0],bbox2[0])
	maxy = max(bbox1[1],bbox2[1])
	intersection = (maxx-minx)*(maxy-miny)
	return intersection/(bbox1[2]*bbox1[3])

def is_valid_multi_IOU(bboxs1, bboxs2, threshold):
	for bbox1 in bboxs1:
		max_iou = 0
		for bbox2 in bboxs2:
			# max_iou = max(max_iou,get_IOU(bbox1,bbox2))
			max_iou = max(max_iou,get_IO1(bbox1,bbox2))
		if max_iou <= threshold:
			return False
	return True

(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
	help="path to input video")
ap.add_argument("-o", "--output", required=True,
	help="path to output video")
ap.add_argument("-y", "--yolo", required=True,
	help="base path to YOLO directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
	help="threshold when applyong non-maxima suppression")
args = vars(ap.parse_args())

# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")

# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])

# load our YOLO object detector trained on COCO dataset (80 classes)
# and determine only the *output* layer names that we need from YOLO
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# initialize the video stream, pointer to output video file, and
# frame dimensions
vs = cv2.VideoCapture(args["input"])
writer = None
(W, H) = (None, None)

# try to determine the total number of frames in the video file
try:
	prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
		else cv2.CAP_PROP_FRAME_COUNT
	total = int(vs.get(prop))
	print("[INFO] {} total frames in video".format(total))

# an error occurred while trying to determine the total
# number of frames in the video file
except:
	print("[INFO] could not determine # of frames in video")
	print("[INFO] no approx. completion time can be provided")
	total = -1

tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
tracker_type = tracker_types[7]
# tracker = initializeTracker(tracker_type,minor_ver)

multi_tracker = cv2.MultiTracker_create()

frame_number = 0
ok = False
initialized = False
frames = []
iou_mismatch_idx = []
# loop over frames from the video file stream
while True:
	frame_number += 1
	# read the next frame from the file
	(grabbed, frame) = vs.read()

	# if the frame was not grabbed, then we have reached the end
	# of the stream
	if not grabbed:
		break

	# if the frame dimensions are empty, grab them
	if W is None or H is None:
		(H, W) = frame.shape[:2]

	if frame_number%5 == 0 or not ok or not initialized:
		# construct a blob from the input frame and then perform a forward
		# pass of the YOLO object detector, giving us our bounding boxes
		# and associated probabilities
		blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
			swapRB=True, crop=False)
		net.setInput(blob)
		start = time.time()
		layerOutputs = net.forward(ln)
		end = time.time()

		# initialize our lists of detected bounding boxes, confidences,
		# and class IDs, respectively
		boxes = []
		confidences = []
		classIDs = []

		# loop over each of the layer outputs
		for output in layerOutputs:
			# loop over each of the detections
			for detection in output:
				# extract the class ID and confidence (i.e., probability)
				# of the current object detection
				scores = detection[5:]
				classID = np.argmax(scores)
				confidence = scores[classID]

				# filter out weak predictions by ensuring the detected
				# probability is greater than the minimum probability
				if confidence > args["confidence"]:
					# scale the bounding box coordinates back relative to
					# the size of the image, keeping in mind that YOLO
					# actually returns the center (x, y)-coordinates of
					# the bounding box followed by the boxes' width and
					# height
					box = detection[0:4] * np.array([W, H, W, H])
					(centerX, centerY, width, height) = box.astype("int")

					# use the center (x, y)-coordinates to derive the top
					# and and left corner of the bounding box
					x = int(centerX - (width / 2))
					y = int(centerY - (height / 2))

					# update our list of bounding box coordinates,
					# confidences, and class IDs
					boxes.append([x, y, int(width), int(height)])
					confidences.append(float(confidence))
					classIDs.append(classID)

		# apply non-maxima suppression to suppress weak, overlapping
		# bounding boxes
		idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
			args["threshold"])

		face_boxes = []
		# ensure at least one detection exists
		if len(idxs) > 0:
			# loop over the indexes we are keeping
			for i in idxs.flatten():
				# extract the bounding box coordinates
				(x, y) = (boxes[i][0], boxes[i][1])
				(w, h) = (boxes[i][2], boxes[i][3])

				x = max(0,x)
				y = max(0,y)
				text = "{}: {:.4f}".format(LABELS[classIDs[i]],
					confidences[i])
				if 'face' in text.lower():
					# draw a bounding box rectangle and label on the frame
					sub_face = frame[y:y+h, x:x+w]
					# apply a gaussian blur on this new recangle image
					sub_face = cv2.GaussianBlur(sub_face,(15, 15), 30)
					# merge this blurry rectangle to our final image
					frame[y:y+sub_face.shape[0], x:x+sub_face.shape[1]] = sub_face

					color = [int(c) for c in COLORS[classIDs[i]]]
					cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
					cv2.putText(frame, text, (x, y - 5),
						cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
					xt = max(x - 0.3*w,0)
					yt = max(y - 0.3*h,0)
					wt = 1.6*w
					ht = 1.6*h
					if not initialized:
						multi_tracker.add(initializeTracker(tracker_type,minor_ver),frame,(xt,yt,wt,ht))
						# tracker.init(frame, (xt,yt,wt,ht))
					else:
						face_boxes.append(boxes[i])
		if initialized:
			ok, bboxs = multi_tracker.update(frame)
			if not is_valid_multi_IOU(face_boxes,bboxs,0.95):
				print("IOU Failed")
				iou_mismatch_idx.append(frame_number-1)
				multi_tracker = cv2.MultiTracker_create()
				for fb in face_boxes:
					fb[0] = max(fb[0] - 0.3*fb[2],0)
					fb[1] = max(fb[1] - 0.3*fb[3],0)
					fb[2] = 1.6*fb[2]
					fb[3] = 1.6*fb[3]
					multi_tracker.add(initializeTracker(tracker_type,minor_ver),frame,(fb[0],fb[1],fb[2],fb[3]))
					# ok = False
					p1 = (int(fb[0]), int(fb[1]))
					p2 = (int(fb[0] + fb[2]), int(fb[1] + fb[3]))
					cv2.rectangle(frame, p1, p2, (0,255,0), 2, 1)
				for bbox in bboxs:
					p1 = (int(max(0,bbox[0])), int(max(0,bbox[1])))
					p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
					cv2.rectangle(frame, p1, p2, (0,0,255), 2, 1)
			else:
				for bbox in bboxs:
					p1 = (int(max(0,bbox[0])), int(max(0,bbox[1])))
					p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
					cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
		initialized = True
				# if ok:
				# 	# Tracking success
				# 	p1 = (int(bbox[0]), int(bbox[1]))
				# 	p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
				# 	sub_face = frame[p1[1]:p2[1], p1[0]:p2[0]]
				# 	# apply a gaussian blur on this new recangle image
				# 	sub_face = cv2.GaussianBlur(sub_face,(15, 15), 30)
				# 	# merge this blurry rectangle to our final image
				# 	frame[p1[1]:p1[1]+sub_face.shape[0], p1[0]:p1[0]+sub_face.shape[1]] = sub_face
				# 	cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
				# else :
				# 	# Tracking failure
				# 	cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
	else:
		ok, bboxs = multi_tracker.update(frame)
		if ok:
			# Tracking success
			for bbox in bboxs:
				p1 = (int(max(0,bbox[0])), int(max(0,bbox[1])))
				p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
				# print(p1,p2,frame.shape)
				sub_face = frame[p1[1]:p2[1], p1[0]:p2[0]]
				# apply a gaussian blur on this new recangle image
				sub_face = cv2.GaussianBlur(sub_face,(15, 15), 30)
				# merge this blurry rectangle to our final image
				frame[p1[1]:p1[1]+sub_face.shape[0], p1[0]:p1[0]+sub_face.shape[1]] = sub_face
				cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
	frames.append(frame)
	print("Frame number:",frame_number)

print("Processing IOU mismatch cases")
e_index = 0
for imi in iou_mismatch_idx:
	s_index = max(0,imi-4,e_index)
	e_index = max(imi,s_index+1)
	for ind in range(s_index,e_index):
		print("Frame number:",ind)
		frame = frames[ind]
		blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
		swapRB=True, crop=False)
		net.setInput(blob)
		start = time.time()
		layerOutputs = net.forward(ln)
		end = time.time()

		# initialize our lists of detected bounding boxes, confidences,
		# and class IDs, respectively
		boxes = []
		confidences = []
		classIDs = []

		# loop over each of the layer outputs
		for output in layerOutputs:
			# loop over each of the detections
			for detection in output:
				# extract the class ID and confidence (i.e., probability)
				# of the current object detection
				scores = detection[5:]
				classID = np.argmax(scores)
				confidence = scores[classID]

				# filter out weak predictions by ensuring the detected
				# probability is greater than the minimum probability
				if confidence > args["confidence"]:
					# scale the bounding box coordinates back relative to
					# the size of the image, keeping in mind that YOLO
					# actually returns the center (x, y)-coordinates of
					# the bounding box followed by the boxes' width and
					# height
					box = detection[0:4] * np.array([W, H, W, H])
					(centerX, centerY, width, height) = box.astype("int")

					# use the center (x, y)-coordinates to derive the top
					# and and left corner of the bounding box
					x = int(centerX - (width / 2))
					y = int(centerY - (height / 2))

					# update our list of bounding box coordinates,
					# confidences, and class IDs
					boxes.append([x, y, int(width), int(height)])
					confidences.append(float(confidence))
					classIDs.append(classID)

		# apply non-maxima suppression to suppress weak, overlapping
		# bounding boxes
		idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
			args["threshold"])

		# ensure at least one detection exists
		if len(idxs) > 0:
			# loop over the indexes we are keeping
			for i in idxs.flatten():
				# extract the bounding box coordinates
				(x, y) = (boxes[i][0], boxes[i][1])
				(w, h) = (boxes[i][2], boxes[i][3])

				x = max(0,x)
				y = max(0,y)
				text = "{}: {:.4f}".format(LABELS[classIDs[i]],
					confidences[i])
				if 'face' in text.lower():
					# draw a bounding box rectangle and label on the frame
					sub_face = frame[y:y+h, x:x+w]
					# apply a gaussian blur on this new recangle image
					sub_face = cv2.GaussianBlur(sub_face,(15, 15), 30)
					# merge this blurry rectangle to our final image
					frame[y:y+sub_face.shape[0], x:x+sub_face.shape[1]] = sub_face

					color = [int(c) for c in COLORS[classIDs[i]]]
					cv2.rectangle(frame, (x, y), (x + w, y + h), (255,255,255), 2)
					cv2.putText(frame, text, (x, y - 5),
						cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
		frames[ind] = frame

for frame in frames:
	# check if the video writer is None
	if writer is None:
		# initialize our video writer
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(args["output"], fourcc, 30,
			(frame.shape[1], frame.shape[0]), True)

		# some information on processing single frame
		if total > 0:
			elap = (end - start)
			print("[INFO] single frame took {:.4f} seconds".format(elap))
			print("[INFO] estimated total time to finish: {:.4f}".format(
				elap * total))
	# write the output frame to disk
	writer.write(frame)

# release the file pointers
print("[INFO] cleaning up...")
writer.release()
vs.release()