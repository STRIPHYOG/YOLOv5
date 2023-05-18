# import required libraries
import cv2
import torch
import numpy as np
# set the path to the pre-trained YOLOv5 model
path1='A:/project/yolov5safetyhelmet-main/best.pt'
# load the YOLOv5 model using torch.hub
model1 = torch.hub.load('ultralytics/yolov5', 'custom',path1, force_reload=True)
# open the input video file using OpenCV VideoCapture
cap1=cv2.VideoCapture('helmet.mp4')
# initialize a variable to count the frames processed
count=0
# loop through the video frames until there are no more frames left
while True:
    # read the next frame from the video file
    ret,frame=cap.read()
    # if there are no more frames, exit the loop
    if not ret:
        break
  # skip every third frame to reduce the processing load
    count += 1
    if count % 3 != 0:
        continue
    # resize the frame to the required size for the YOLOv5 model
    frame=cv2.resize(frame,(860,660))
    # use the YOLOv5 model to detect objects in the frame
    results=model(frame)
    # render the results onto the frame and convert it to a numpy array
    frame=np.squeeze(results.render())
    # display the frame in a window titled "My DataSet"
    cv2.imshow("FRAME",frame)
    # if the user presses the "Esc" key, exit the loop
    if cv2.waitKey(1)&0xFF==27:
        break
# release the video capture object and destroy any OpenCV windows
cap1.release()
cv2.destroyAllWindows()
