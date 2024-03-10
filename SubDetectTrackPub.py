#!/usr/bin/env
import speech_recognition as sr
import pyttsx3 as tts
import rospy
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String
import numpy as np
import ros_numpy
from PIL import Image
from ultralytics import YOLO
import cv2

#CREATE GLOBAL YOLO MODEL SO TRACKING CAN PERSIST BETWEEN FRAMES
global model
model = YOLO('/home/ubuntu/python_catkin/runs/detect/train6/weights/best.pt')

def scale_to_255(a, min, max, dtype=np.uint8):

    return (((a - min) / float(max - min)) * 255).astype(dtype)

def point_cloud_2_birdseye(points,
                           res=0.01,
                           side_range=(-10., 10.),  # left-most to right-most
                           fwd_range = (-10., 10.), # back-most to forward-most
                           height_range=(-2., 2.),  # bottom-most to upper-most
                           ):

    # EXTRACT THE POINTS FOR EACH AXIS
    x_points = np.array(list(zip(*points))[0])
    y_points = np.array(list(zip(*points))[1])
    z_points = np.array(list(zip(*points))[2])

    # FILTER - To return only indices of points within desired cube
    # Three filters for: Front-to-back, side-to-side, and height ranges
    # Note left side is positive y axis in LIDAR coordinates
    f_filt = np.logical_and(x_points > fwd_range[0], x_points < fwd_range[1])
    s_filt = np.logical_and((y_points > -side_range[1]), (y_points < -side_range[0]))
    filter = np.logical_and(f_filt, s_filt)
    indices = np.argwhere(filter).flatten()

    # KEEPERS
    x_points = x_points[indices]
    y_points = y_points[indices]
    z_points = z_points[indices]

    # CONVERT TO PIXEL POSITION VALUES - Based on resolution
    x_img = (-y_points / res).astype(np.int32)  # x axis is -y in LIDAR
    y_img = (-x_points / res).astype(np.int32)  # y axis is -x in LIDAR

    # SHIFT PIXELS TO HAVE MINIMUM BE (0,0)
    # floor & ceil used to prevent anything being rounded to below 0 after shift
    x_img -= int(np.floor(side_range[0] / res))
    #print("xpoints after %d", len(x_img))
    y_img += int(np.ceil(fwd_range[1] / res))

    # CLIP HEIGHT VALUES - to between min and max heights
    pixel_values = np.clip(a=z_points,
                           a_min=height_range[0],
                           a_max=height_range[1])

    # RESCALE THE HEIGHT VALUES - to be between the range 0-255
    pixel_values = scale_to_255(pixel_values,
                                min=height_range[0],
                                max=height_range[1])

    # INITIALIZE EMPTY ARRAY - of the dimensions we want
    x_max = 1 + int((side_range[1] - side_range[0]) / res)
    y_max = 1 + int((fwd_range[1] - fwd_range[0]) / res)
    im = np.zeros([y_max, x_max], dtype=np.uint8)

    # FILL PIXEL VALUES IN IMAGE ARRAY
    im[y_img, x_img] = pixel_values

    return im

def publisher(msg):
    pub = rospy.Publisher('python_output', String, queue_size=20)
    pub.publish(msg)

def create_topic_data(data, follow, follow_id):
    if (data):
        x = data[follow_id][0] #get x value from selected track
        y = data[follow_id][1] #get y value from selected track
        #print("x = {}, y = {}".format(x, y))
        x = x/4096 #convert x value from pixels to meters
        y = y/4096 #convert y value from pixels to meters
        output = "x = {}, y = {}, follow = {}".format(x, y, follow)
        #print("x = {}, y = {}".format(x, y))
        publisher(output)

def callback(data):
    follow = 0

    #TURN DATA INTO NUMPY ARRAY
    arr = ros_numpy.numpify(data)

    #CREATE BIRDS EYE VIEW 2D IMAGE
    img = point_cloud_2_birdseye(arr)
    img = Image.fromarray(img)

    #RUN OBJECT DETECTION AND TRACKING ON IMAGE
    results = model.track(img, tracker="bytetrack.yaml", persist=True)
    boxes = results[0].boxes.xywh.cpu() #get values of bounding boxes
    ids = results[0].boxes.id.int().cpu().tolist() #get ids of bounding boxes

    #CREATE FRAME TO BE DISPLAYED
    annotated_frame = results[0].plot()

    #FIND CLOSEST TRACK TO LIDAR
    boxes = boxes.tolist()
    follow_id = 0
    if (len(boxes) > 1):
        for index, item in enumerate(boxes):
            if ((item[0]+item[1])/2 < (boxes[follow_id][0]+boxes[follow_id][1])/2):
                follow_id = index

    #SEND DATA FOR OUTPUT PROCESSING
    create_topic_data(boxes, follow, follow_id)

    #DISPLAY FRAME AND UPDATE FOR EACH NEW FRAME
    cv2.imshow("YOLOv8 Tracking", annotated_frame)
    cv2.waitKey(1)
    

def listener():
    #INITIALIZE AND RUN ROS SUBSCRIBER UNTIL NO MORE DATA IS PRESENT
    rospy.init_node('PubSub', anonymous=True)
    rospy.Subscriber('/cloud_all_fields_fullframe', PointCloud2, callback)
    rospy.spin()
    

if __name__ == '__main__':
    try:
        listener()
    except rospy.ROSInterruptException:
        pass