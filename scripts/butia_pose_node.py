#!/usr/bin/env python3
import rospy
import ros_numpy

from jetson_inference import poseNet
from jetson_utils import cudaFromNumpy
from butia_vision_bridge import VisionSynchronizer

class ButiaPose():
    
    def __init__(self):
        self.img = None
        self.net = None

        self._readParameters()
        self._loadNetwork()
        VisionSynchronizer.syncSubscribers(self.source_topic_dict, self.callback)


        self.run()
        
    def run(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            rate.sleep()

    def _loadNetwork(self):
        self.net = poseNet(self.network)

    def _readParameters(self):
        self.source_topic_dict = {"image_rgb": "/camera/color/image_raw"}

        self.network = "resnet18-body"

    def callback(self, *args):
        img = None
        if len(args):
            img = args[0]
        cv_img = ros_numpy.numpify(img)
        cudaImage = cudaFromNumpy(cv_img)

        poses = self.net.Process(cudaImage, overlay="keypoints")
        
        print("Was detected {:d} people".format(len(poses)))

if __name__ == "__main__":
    rospy.init_node("pose_node")

    bp =  ButiaPose()

