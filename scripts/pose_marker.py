#!/usr/bin/env python
import rospy
from visualization_msgs.msg import Marker, MarkerArray
from butia_vision_msgs.msg import Frame, BodyPart, Pixel, Person

pub = rospy.Publisher("/butia_vision/pose/markers",MarkerArray, queue_size=1)

def callback(frame):
    global pub
    i = 0
    markers = MarkerArray()
    for person in frame.persons:

       for part in person.bodyParts:

           if part.score >= 0.8 and part.point != None:
               marker = Marker()
               marker.header = frame.header
               marker.type = 2
               marker.id = i
               marker.color.r = 0.0
               marker.color.g = 1.0
               marker.color.b = 0.0
               marker.color.a = 0.8
               marker.scale.x = 0.05
               marker.scale.y = 0.05
               marker.scale.z = 0.05

               marker.pose.position.x = part.point.x
               marker.pose.position.y = part.point.y
               marker.pose.position.z = part.point.z
               marker.pose.orientation.x = 0.0
               marker.pose.orientation.y = 0.0
               marker.pose.orientation.z = 0.0
               marker.pose.orientation.w = 1.0
               marker.lifetime = rospy.Duration(1.0)
               print(marker)
               markers.markers.append(marker)
               i += 1
    pub.publish(markers)





if __name__ == "__main__":
    rospy.init_node("pose_marker")
    sub = rospy.Subscriber("/butia_vision/pose", Frame, callback)
    rospy.spin()
    
	
