#!/usr/bin/env python3
import rospy
import ros_numpy

from ultralytics import YOLO

from butia_vision_bridge import VisionSynchronizer, VisionBridge
from butia_vision_msgs.msg import Frame, BodyPart, Pixel, Person

from sensor_msgs.msg import PointCloud2, Image
from geometry_msgs.msg import Point

import copy
import cv2

#roslaunch realsense2_camera rs_camera.launch depth_width:=424 depth_height:=240 color_width:=424 color_height:=240 depth_fps:=15 color:=15 filters:=pointcloud ordered_pc:=true color_fps:=15
class ButiaPose():

    open_pose_map = { "nose": 0,
                     "neck" : 1,
                     "right_shoulder" : 2,
                     "right_elbow" : 3,
                     "right_wrist": 4,
                     "left_shoulder" : 5,
                     "left_elbow": 6,
                     "left_wrist" : 7,
                     "MidHip" : 8,
                     "right_hip": 9,
                     "right_knee" : 10,
                     "right_ankle" : 11,
                     "left_hip": 12,
                     "left_knee" : 13,
                     "left_ankle" : 14,
                     "right_eye" : 15,
                     "left_eye" : 16,
                     "right_ear" : 17,
                     "left_ear" : 18,
                     "LBigToe" : 19,
                     "LSmallToe": 20,
                     "LHeel": 21,
                     "RBigToe": 22,
                     "RSmallToe": 23,
                     "RHeel":24,
                     "Background":25}
        

    yolo_map = (
       "nose",
       "left_eye",
       "right_eye",
       "left_ear",
       "right_ear",
       "left_shoulder",
       "right_shoulder",
       "left_elbow",
       "right_elbow",
       "left_wrist",
       "right_wrist",
       "left_hip",
       "right_hip",
       "left_knee",
       "right_knee",
       "left_ankle",
       "right_ankle",
       )
    
    def __init__(self):
        self.img = None
        self.net = None

        #self.out = videoOutput("display://0")

        self._readParameters()
        self._loadNetwork()

        VisionSynchronizer.syncSubscribers(self.source_topic_dict, self.callback)
        self._pub = rospy.Publisher("/butia_vision/pose",Frame,queue_size=10)
        self._pubDebug = rospy.Publisher("/butia_vision/pose/image", Image, queue_size=10)

        self.run()
        
    def run(self):
        rate = rospy.Rate(1)
        while not rospy.is_shutdown():
            rate.sleep()

    def _loadNetwork(self):
        self.net = YOLO(self.network)

    def _readParameters(self):
        self.source_topic_dict = {
		"points":"/camera/depth/color/points",
		"image_rgb":"/camera/color/image_raw"
                }

        self.network = "yolov8-pose.pt"

    def callback(self, *args):
        #print("CHEGUEI AQUI")

        # Init the Frame message and write you header
        frame = Frame()
        frame.header.stamp = rospy.Time.now()

        # Init the img and points variable
        img = None
        points = None

        # To verify if exists messages being published in topics (self.source_topic_dict)
        if len(args):
            img = args[0]
            points = args[1]
            frame.header = points.header

        # Convert the message to numpy array
        cv_img = ros_numpy.numpify(img)
        #print(cv_img.shape)
        cv_points = ros_numpy.numpify(points)

        results = self.net(cv_img,save=False)
        annotated_frame = results[0].plot()
        for i in range(len(results[0].boxes)):
            if results[0].boxes.conf[i].item() > 0.8:
                pose  = Person()
                pose.bodyParts = [BodyPart() for _ in range(len(self.open_pose_map.values()))]
                pose.leftHandParts = [BodyPart() for _ in range(21)]
                pose.rightHandParts = [BodyPart() for _ in range(21)]
                pose.faceParts = [BodyPart() for _ in range(70)]

                for idx, kpt in enumerate(results[0].keypoints[i]):
                    if kpt[2] > 0.8:
                        pixel = Pixel()
                        pixel.x = kpt[0]
                        pixel.y = kpt[1]
                        i = self.open_pose_map[self.yolo_map[idx]]
                        #ADD SCORE
                        pose.bodyParts[i].pixel = pixel
                        pose.bodyParts[i].score = kpt[2]
                        pose.bodyParts[i].point = getCloudPointFromImage(kpt[0],kpt[1], cv_points)
                
                frame.persons.append(pose)
        self._pub(frame)
        self._pubDebug(ros_numpy.msgify(Image, annotated_frame))
  
def getCloudPointFromImage(x, y, points) -> Point:
        x_3D, y_3D, z_3D, p3d = points[int(x), int(y)]
        #print(f"{x} {y} {z}")
        point  = Point()
        point.x = x_3D
        point.y = y_3D
        point.z = z_3D
        #print(f"{x} and {y} : {point}")
        return point

if __name__ == "__main__":
    rospy.init_node("pose_node")

    bp =  ButiaPose()

