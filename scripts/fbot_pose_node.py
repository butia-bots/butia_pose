#!/usr/bin/env python
import rospy
import ros_numpy

from ultralytics import YOLO

from butia_vision_bridge import VisionSynchronizer, VisionBridge
from butia_vision_msgs.msg import Frame, BodyPart, Pixel, Person

from sensor_msgs.msg import PointCloud2, Image
from geometry_msgs.msg import Point

import copy
import cv2
import open3d as o3d
import numpy as np
from math import pow

#roslaunch realsense2_camera rs_camera.launch depth_width:=424 depth_height:=240 color_width:=424 color_height:=240 depth_fps:=15 color:=15 filters:=pointcloud ordered_pc:=true color_fps:=15
already_printed = False
class FBotPose():

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

        VisionSynchronizer.syncSubscribers(self.source_topic_dict, self.callback, slop=0.5)
        self._pub = rospy.Publisher("/butia_vision/pose",Frame,queue_size=1)
        self._pubDebug = rospy.Publisher("/butia_vision/pose/image", Image, queue_size=1)
        self._VOXEL_SIZE = 0.03
        self._EPS = 2 * self._VOXEL_SIZE * np.sqrt(3)

        self.run()
        
    def run(self):
        rate = rospy.Rate(1)
        while not rospy.is_shutdown():
            rate.sleep()

    def _loadNetwork(self):
        self.net = YOLO(self.network)

    def _readParameters(self):
        self.source_topic_dict = {
		"points":"/kinect2/qhd/points",
		"image_rgb":"/kinect2/qhd/image_color_rect"
                }

        self.network = "yolov8n-pose.pt"

    def callback(self, *args):
        # Init the Frame message and write you header
        frame = Frame()
        #frame.header.stamp = rospy.Time.now()

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
        xyz, rgb = VisionBridge.pointCloud2XYZRGBtoArrays(points)
        array_point_cloud = np.append(xyz, rgb, axis=2)

        results = self.net(cv_img,save=False)
        annotated_frame = results[0].plot()
        self._pubDebug.publish(ros_numpy.msgify(Image, annotated_frame,encoding=img.encoding))
        for i in range(len(results[0].boxes)):
            if results[0].boxes.conf[i].item() > 0.8:
                pose  = Person()
                pose.bodyParts = [BodyPart() for _ in range(len(self.open_pose_map.values()))]
                pose.leftHandParts = [BodyPart() for _ in range(21)]
                pose.rightHandParts = [BodyPart() for _ in range(21)]
                pose.faceParts = [BodyPart() for _ in range(70)]

                box = results[0].boxes[i].xyxy[0].cpu().numpy()
                # print(box)
                # center = cv_points[int((box[1]+box[3])//2), int((box[0]+box[2])//2)]
                # print(getCloudPointFromImage(center[0],center[1],points=cv_points))

                for idx, kpt in enumerate(results[0].keypoints[i]):
                    if kpt[2] > 0.8:
                        pixel = Pixel()
                        pixel.x = kpt[0]
                        pixel.y = kpt[1]
                        i = self.open_pose_map[self.yolo_map[idx]]
                        #ADD SCORE
                        pose.bodyParts[i].pixel = pixel
                        point = self.__imageToPoint(int(kpt[0]),int(kpt[1]), array_point_cloud, box)
                        if point != None:
                            pose.bodyParts[i].point = point
                            pose.bodyParts[i].score = kpt[2]
                        else:
                            pose.bodyParts[i].score = 0
                frame.persons.append(pose)
        self._pub.publish(frame)

    def __imageToPoint(self, x, y, cloud, box):
        pcd = o3d.geometry.PointCloud()
        # cloud = np.asarray(cloud)
        # print(type(cloud[0,0]))
        sub_cloud = cloud[max(int(box[1]),y-5):min(int(box[3]),y+5), max(int(box[0]),x-5):min(int(box[2]),x+5),:]
        # print(sub_cloud.shape)
        sub_cloud = sub_cloud.reshape((-1,3))
        pcd.points = o3d.utility.Vector3dVector(sub_cloud)
        pcd = pcd.remove_non_finite_points()
        pcd = pcd.voxel_down_sample(self._VOXEL_SIZE)
        pointsn = len(pcd.points)
        labels_array = np.asarray(pcd.cluster_dbscan(eps=0.03*1.2,min_points=pointsn//4))
        labels, count = np.unique(labels_array, return_counts=True)
        clusters = []
        for label in labels:
            if label < 0:
                continue
            clusters.append([])
            for label_id, point in zip(labels_array, pcd.points):
                if label_id == label:
                    clusters[label].append(point)

        if len(clusters) == 0:
            return None
        
        print("*"*10)

        zs = []
        for cluster in clusters:
            # print(np.array(cluster))
            print("-"*10)
            vals = []
            for point in cluster:
                vals.append(point[2])
            
            zs.append(np.mean(vals))

            # print(cluster)
        # print(min(zs))
        index = zs.index(min(zs))
        print(index)

        valsx = []
        valsy = []
        for point in clusters[index]:
            valsx.append(point[0])
            valsy.append(point[1])
        
        x = np.mean(valsx)
        y = np.mean(valsy)

        point = Point()
        point.x = x
        point.y = y
        point.z = zs[index]
        print(point)

        return point



# def getCloudPointFromImage(x, y, points, center):
#         KERNEL_SIZE = 10
#         global already_printed
#         x = int(x)
#         y = int(y)
#         candidates_filtered = []
#         x_3D = None
#         y_3D = None
#         z_3D = None
#         if center != None:
#             candidates = points[
#                 max(0,y-KERNEL_SIZE):min(points.shape[0]-1,y+KERNEL_SIZE+1),
#                 max(0,x-KERNEL_SIZE):min(points.shape[0]-1,x+KERNEL_SIZE+1)
#                 ]
#             # print(center)
#             for p in candidates:
#                 for xi,yi,zi,p3d in p:
#                     # print(f"{xi} {yi} {zi}")
#                     if np.sqrt(pow(xi - center[0],2) + pow(yi - center[1],2) + pow(zi - center[2],2)) < 2.0:
#                         candidates_filtered.append([xi,yi,zi])
            
#             candidates_filtered.sort(key=lambda p: p[2])
#             size = len(candidates_filtered)
#             if size > 0:
#                 # print(candidates1[(size//2)-1])
#                 x_3D, y_3D, z_3D = candidates_filtered[(size//2)-1]
#                 # print(f"{x_3D} {y_3D} {z_3D}")
        
#         if x_3D == None:
#             x_3D, y_3D, z_3D, p3d = points[min(int(y), points.shape[0]-1), min(int(x), points.shape[1]-1)]
#         #print(f"{x} {y} {z}")
#         point  = Point()
#         point.x = x_3D
#         point.y = y_3D
#         point.z = z_3D
#         #print(f"{x} and {y} : {point}")
#         return point

if __name__ == "__main__":
    rospy.init_node("pose_node")

    bp =  FBotPose()

