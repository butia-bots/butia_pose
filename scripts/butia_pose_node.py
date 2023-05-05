#!/usr/bin/env python3
import rospy
import ros_numpy

from jetson_inference import poseNet
from jetson_utils import cudaFromNumpy, videoOutput

from butia_vision_bridge import VisionSynchronizer, VisionBridge
from butia_vision_msgs.msg import Frame, BodyPart, Pixel, Person

from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import Point

import copy

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
        

#    body_parts_map = (
#        "nose",
#        "left_eye",
#        "right_eye",
#        "left_ear",
#        "right_ear",
#        "left_shoulder",
#        "right_shoulder",
#        "left_elbow",
#        "right_elbow",
#        "left_wrist",
#        "right_wrist",
#        "left_hip",
#        "right_hip",
#        "left_knee",
#        "right_knee",
#        "left_ankle",
#        "right_ankle",
#        "neck"
#        )
    
    def __init__(self):
        self.img = None
        self.net = None

        #self.out = videoOutput("display://0")

        self._readParameters()
        self._loadNetwork()

        VisionSynchronizer.syncSubscribers(self.source_topic_dict, self.callback)
        self._pub = rospy.Publisher("/butia_vision/pose",Frame,queue_size=10)

        self.run()
        
    def run(self):
        rate = rospy.Rate(1)
        while not rospy.is_shutdown():
            rate.sleep()

    def _loadNetwork(self):
        self.net = poseNet(self.network)
        self.net.SetKeypointScale(1)

    def _readParameters(self):
        self.source_topic_dict = {
		"points":"/camera/depth/color/points",
		"image_rgb":"/camera/color/image_raw"
                }

        self.network = "resnet18-body"

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

        # Create cudaImage object from cv_img
        cuda_image = cudaFromNumpy(cv_img)

        # Get the poses (how many people) are being detected
        #print(cv_img.shape[0])
        #print(cv_img.shape[1])  
        poses = self.net.Process(cuda_image, overlay="keypoints")
        #print(poses[0].x)
        #self.out.Render(cuda_image)
        #self.out.SetStatus("Preview")

        # If any person was detected
        if len(poses):
            nb_persons = len(poses)
            bodypart_count = len(poses)
        else:
            nb_persons = 0
            bodypart_count = 0
        pc = ros_numpy.numpify(points)

        # Create the Persons messages for each person detected
#        print(self.net.GetKeypointScale())
        if nb_persons != 0:
            print(len(poses))
            for person in poses:
                #print(person.Keypoints)                
                pose  = Person()
                pose.bodyParts = [BodyPart() for _ in range(len(self.open_pose_map.values()))]
                pose.leftHandParts = [BodyPart() for _ in range(21)]
                pose.rightHandParts = [BodyPart() for _ in range(21)]
                pose.faceParts = [BodyPart() for _ in range(70)]

                for part in person.Keypoints:
                    pixel = Pixel()
                    i = self.open_pose_map[self.net.GetKeypointName(part.ID)]
                    pixel.x = part.x
                    pixel.y = part.y
                    pose.bodyParts[i].pixel = pixel
                    if (0 <= pixel.x <= pc.shape[0]) and (0 <= pixel.y <= pc.shape[1]):
                        pose.bodyParts[i].point = getCloudPointFromImage(pixel.x, pixel.y,pc)
                        #print(f"On_pose:{pose.bodyParts[i].point}")
                        pose.bodyParts[i].score = 1.0
                    else:
                        pose.bodyParts[i].score = 0.0
                    
		

		##################bodyparts###############
#               for i, name in self.open_pose_map.items():
#                  pixel = Pixel()
#                  if name in self.body_parts_map:
#                       #print(name)
#                       print(self.net.FindKeypointID(name))
#                       print(len(person.Keypoints))
#                       part = person.Keypoints[self.net.FindKeypointID(name)]
#                       pixel.x = part.x
#                       pixel.y = part.y
#                       print(f"PIXELS {pixel.x} {pixel.y}")
#                       pose.bodyParts[i].pixel = pixel
#                       if (0 <= pixel.x <= pc.shape[0]) and (0 <= pixel.y <= pc.shape[1]):
#                           pose.bodyParts[i].point = getCloudPointFromImage(pixel.x, pixel.y,pc)
#                           pose.bodyParts[i].score = 1.0
#                       else:
#                           pose.bodyParts[i].score = 0.0
#                        
#                   else:
#                        pixel.x = -1
#                        pixel.y = -1
#                        pose.bodyParts[i].pixel = pixel
#                        point = Point()
#                        point.x = -1
#                        point.y = -1
#                        point.z = -1
#                        pose.bodyParts[i].point = getCloudPointFromImage(pixel.x, pixel.y,pc)
#                        pose.bodyParts[i].score = 0.0
		#################left hand#################
                for hand_part in pose.leftHandParts:
                        pixel = Pixel()
                        pixel.x = -1
                        pixel.y = -1
                        hand_part.pixel = pixel
                        point = Point()
                        point.x = -1
                        point.y = -1
                        point.z = -1
                        hand_part.score = 0.0
		#################right hand#################
                for hand_part in pose.rightHandParts:
                        pixel = Pixel()
                        pixel.x = -1
                        pixel.y = -1
                        hand_part.pixel = pixel
                        point = Point()
                        point.x = -1
                        point.y = -1
                        point.z = -1
                        hand_part.score = 0.0
		#################face      #################
                for face_part in pose.faceParts:
                        pixel = Pixel()
                        pixel.x = -1
                        pixel.y = -1
                        face_part.pixel = pixel
                        point = Point()
                        point.x = -1
                        point.y = -1
                        point.z = -1
                        face_part.score = 0.0
                frame.persons.append(pose)
        print(len(frame.persons))
        self._pub.publish(frame)
        #print(len(pc.data))	

def getCloudPointFromImage(x, y, points) -> Point:
        #np_points = ros_numpy.numpify(points)
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

