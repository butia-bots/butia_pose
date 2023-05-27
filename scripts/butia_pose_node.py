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
    
    colors = [(255,0,0),(0,255,0),(0,125,255),(255,255,0),(255,0,255),(0,0,255)]
    
    def __init__(self):
        self.img = None
        self.net = None
        self._readParameters()
        self._loadNetwork()

        VisionSynchronizer.syncSubscribers(self.source_topic_dict, self.callback, slop=self._slop)
        self._pub = rospy.Publisher(self._pubTopic, Frame, queue_size=self._queue)
        self._pubDebug = rospy.Publisher(self._pubDebugTopic, Image, queue_size=self._queue)
        self._pubDebugCluster = rospy.Publisher("/butia_vision/pose/clusters", PointCloud2, queue_size=self._queue)
        # self._EPS = 2 * self._VOXEL_SIZE * np.sqrt(3)
        self._EPS = self._VOXEL_SIZE * 1.2
        rospy.logerr(self._VOXEL_SIZE)

        self.run()
        
    def run(self):
        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            rate.sleep()

    def _loadNetwork(self):
        self.net = YOLO(self.network)

    def _readParameters(self):
        self.source_topic_dict = {
		"points": rospy.get_param("~subscribers/points","/kinect2/qhd/points"),
		"image_rgb": rospy.get_param("~subscribers/image_rgb", "/kinect2/qhd/image_color_rect")
                }

        self.network = rospy.get_param("~model_file", "yolov8n-pose.pt")
        self._pubTopic = rospy.get_param("~publishers/pose_detection/data","/butia_vision/pose")
        self._pubDebugTopic = rospy.get_param("~publishers/pose_detection/debug_image","/butia_vision/pose/image")
        self._queue = rospy.get_param("~subscribers/queue_size", 1)
        self._slop = rospy.get_param("~subscribers/slop",0.5)
        self._VOXEL_SIZE = rospy.get_param("~voxel_size",0.03)
        self._threshold = rospy.get_param("~threshold",0.8)

    def callback(self, *args):
        frame = Frame()
        img = None
        points = None
        if len(args):
            img = args[0]
            points = args[1]
            frame.header = points.header
        cv_img = ros_numpy.numpify(img)
        xyz, rgb = VisionBridge.pointCloud2XYZRGBtoArrays(points)
        array_point_cloud = np.append(xyz, rgb, axis=2)

        results = self.net(cv_img,save=False)
        annotated_frame = results[0].plot()
        self._pubDebug.publish(ros_numpy.msgify(Image, annotated_frame,encoding=img.encoding))

        self._clusters_points = np.array([[0,0,0]])
        self._clusters_points_colors = np.array([[0,0,0]])

        for i in range(len(results[0].boxes)):
            if results[0].boxes.conf[i].item() > self._threshold:
                pose  = Person()
                pose.bodyParts = [BodyPart() for _ in range(len(self.open_pose_map.values()))]
                pose.leftHandParts = [BodyPart() for _ in range(21)]
                pose.rightHandParts = [BodyPart() for _ in range(21)]
                pose.faceParts = [BodyPart() for _ in range(70)]

                box = results[0].boxes[i].xyxy[0].cpu().numpy()
                for idx, kpt in enumerate(results[0].keypoints[i]):
                    if kpt[2] > 0.8:
                        pixel = Pixel()
                        pixel.x = kpt[0]
                        pixel.y = kpt[1]
                        i = self.open_pose_map[self.yolo_map[idx]]
                        #ADD SCORE
                        pose.bodyParts[i].pixel = pixel
                        point = self.__imageToPoint(int(kpt[0]),int(kpt[1]), array_point_cloud, box, points.header)
                        if point != None:
                            pose.bodyParts[i].point = point
                            pose.bodyParts[i].score = kpt[2]
                        else:
                            pose.bodyParts[i].score = 0
                frame.persons.append(pose)
        # rospy.logerr(self._clusters_points)
        # rospy.logerr(self._clusters_points_colors)
        self._pubDebugCluster.publish(VisionBridge.arrays2toPointCloud2XYZRGB(self._clusters_points, self._clusters_points_colors, points.header))
        self._pub.publish(frame)

    def __imageToPoint(self, x, y, cloud, box, header):
        pcd = o3d.geometry.PointCloud()
        sub_cloud = cloud[max(int(box[1]),y-5):min(int(box[3]),y+5), max(int(box[0]),x-5):min(int(box[2]),x+5),:]
        sub_cloud = sub_cloud.reshape((-1,3))
        pcd.points = o3d.utility.Vector3dVector(sub_cloud)
        pcd = pcd.remove_non_finite_points()
        if len(pcd.points) == 0:
            rospy.logwarn(f"Key points has no valid 3D points!!!")
        pcd = pcd.voxel_down_sample(self._VOXEL_SIZE)
        pointsn = len(pcd.points)
        labels_array = np.asarray(pcd.cluster_dbscan(eps=0.03*1.2,min_points=pointsn//4))
        labels, count = np.unique(labels_array, return_counts=True)
        clusters = []

        noise_cluster = []
        for label in labels:
            if label < 0:
                continue
            clusters.append([])
            for label_id, point in zip(labels_array, pcd.points):
                if label_id == label:
                    clusters[label].append(point)
                elif label_id == -1:
                    noise_cluster.append(point)

        # rospy.logwarn(noise_cluster)

        # if len(clusters) == 0:
        #     clusters.append(noise_cluster)
        #     noise_cluster = []
        
        zs = []

        # all_points = []
        # all_points_colors = []
        for i, cluster in enumerate(clusters):
            # pc = VisionBridge.arrays2toPointCloud2XYZRGB(cluster, (self.colors[i])*len(cluster),)
            # rospy.logerr(np.asarray(np.repeat([self.colors[i]],repeats=len(cluster),axis=0)))
            self._clusters_points = np.concatenate((self._clusters_points, np.array(cluster)), axis=0)
            self._clusters_points_colors = np.concatenate((self._clusters_points_colors, np.asarray(np.repeat([self.colors[i]],repeats=len(cluster),axis=0))),axis=0)
            print("-"*10)
            vals = []
            for point in cluster:
                vals.append(point[2])
            zs.append(np.mean(vals))
        
        # rospy.logwarn(noise_cluster)
        if len(noise_cluster) != 0:
            self._clusters_points = np.concatenate((self._clusters_points, np.asarray(noise_cluster)),axis=0)
            self._clusters_points_colors = np.concatenate((self._clusters_points_colors, np.asarray(np.repeat([self.colors[-1]],repeats=len(noise_cluster),axis=0))),axis=0)
        # pc_msg = VisionBridge.arrays2toPointCloud2XYZRGB(all_points, all_points_colors, header)
        # self._pubDebugCluster.publish(pc_msg)
        z = 0
        xs = []
        ys = []
        if len(clusters) != 0:
            index = zs.index(min(zs))
            for point in clusters[index]:
                xs.append(point[0])
                ys.append(point[1])
            z = zs[index]
        else:
            zs = []
            xs = []
            ys = []
            for point in noise_cluster:
                zs.append(point[2])
                xs.append(point[0])
                ys.append(point[1])
            z = np.mean(zs)
        
        x = np.mean(xs)
        y = np.mean(ys)

        point = Point()
        point.x = x
        point.y = y
        point.z = z

        return point
if __name__ == "__main__":
    rospy.init_node("pose_node", log_level=rospy.INFO)

    bp =  ButiaPose()

