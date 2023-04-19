#!/usr/bin/env python3
import rospy
import ros_numpy

from jetson_inference import poseNet
from jetson_utils import cudaFromNumpy

from butia_vision_bridge import VisionSynchronizer, VisionBridge
from butia_vision_msgs.msg import Frame, BodyPart, Pixel, Person

from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import Point

import copy

class ButiaPose():

    open_pose_map = { 0: "nose",
                      1: "neck",
                      2: "right_shoulder",
                      3: "right_elbow",
                      4: "right_wrist",
                      5: "left_shoulder",
                      6: "left_elbow",
                      7: "left_wrist",
                      8: "MidHip",
                      9: "right_hip",
                     10: "right_knee",
                     11: "right_ankle",
                     12: "left_hip",
                     13: "left_knee",
                     14: "left_ankle",
                     15: "right_eye",
                     16: "left_eye",
                     17: "right_ear",
                     18: "left_ear",
                     19: "LBigToe",
                     20: "LSmallToe",
                     21: "LHeel",
                     22: "RBigToe",
                     23: "RSmallToe",
                     24: "RHeel",
                     25: "Background"}
        

    body_parts_map = (
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
        "neck"
        )
    
    def __init__(self):
        self.img = None
        self.net = None

        self._readParameters()
        self._loadNetwork()

        VisionSynchronizer.syncSubscribers(self.source_topic_dict, self.callback)

        self.run()
        
    def run(self):
        rate = rospy.Rate(1)
        while not rospy.is_shutdown():
            rate.sleep()

    def _loadNetwork(self):
        self.net = poseNet(self.network)

    def _readParameters(self):
        self.source_topic_dict = {
                "image_rgb": "/camera/color/image_raw",
                "points": "/camera/depth/color/points",
                }

        self.network = "resnet18-body"

    def callback(self, *args):

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

        # Convert the message to numpy array
        cv_img = ros_numpy.numpify(img)
        cv_points = ros_numpy.numpify(points)

        # Create cudaImage object from cv_img
        cuda_image = cudaFromNumpy(cv_img)

        # Get the poses (how many people) are being detected
        poses = self.net.Process(cuda_image, overlay="keypoints")

        # If any person was detected
        if len(poses):
            nb_persons = len(poses)
            bodypart_count = len(poses)
        else:
            nb_persons = 0
            bodypart_count = 0

        # Create the Persons messages for each person detected
        if nb_persons != 0:
            for person in poses:                
                pose  = Person()
                pose.bodyParts = [BodyPart() for _ in range(bodypart_count)]
                pose.leftHandParts = [BodyPart() for _ in range(bodypart_count)]
                pose.rightHandParts = [BodyPart() for _ in range(bodypart_count)]
                for i, name in self.open_pose_map.items():
                    if name in self.body_parts_map():
                        pose.bodyParts[name] = person

            pc = ros_numpy.numpify(points)
            print(len(pc.data))

def getCloudPointFromImage(x : float, y: float, points: PointCloud2) -> Point:
        np_points = ros_numpy.numpify(points)
        x_3D, y_3D, z_3D, _ = np_points[x, y]
        print(f"{x} {y} {z}")
        point  = Point()
        point.x = x_3D
        point.y = y_3D
        point.z = z_3D
        return point

if __name__ == "__main__":
    rospy.init_node("pose_node")

    bp =  ButiaPose()

