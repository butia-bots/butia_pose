#!/usr/bin/env python
import rospy
from visualization_msgs.msg import Marker
from butia_vision_msgs.msg import Frame, BodyPart, Pixel, Person

pub = rospy.Publisher("/butia_vision/pose/markers",Marker, queue_size=10)

def callback(frame):

   global pub
   i = 0
   for person in frame.persons:

       for part in person.bodyParts:

           if part.score >= 0.8:
               marker = Marker()
               marker.header = frame.header
               marker.type = 2
               marker.id = i
               marker.color.r = 0.0
               marker.color.g = 1.0
               marker.color.b = 0.0
               marker.color.a = 0.8
               marker.scale.x = 0.1
               marker.scale.y = 0.1
               marker.scale.z = 0.1

               marker.pose.position.x = part.point.x
               marker.pose.position.y = part.point.y
               marker.pose.position.z = part.point.z
               marker.pose.orientation.x = 0.0
               marker.pose.orientation.y = 0.0
               marker.pose.orientation.z = 0.0
               marker.pose.orientation.w = 1.0
               marker.lifetime = rospy.Duration(10.0)
               print(marker)
               pub.publish(marker)
               i += 1





if __name__ == "__main__":
    rospy.init_node("pose_marker")
    sub = rospy.Subscriber("/butia_vision/pose", Frame, callback)
    rospy.spin()
    
	
