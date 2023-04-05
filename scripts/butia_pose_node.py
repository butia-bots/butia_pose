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
            print("Ola mundo---------------")
            pc = ros_numpy.numpify(points)
            print(len(pc.data))

if __name__ == "__main__":
    rospy.init_node("pose_node")

    bp =  ButiaPose()

