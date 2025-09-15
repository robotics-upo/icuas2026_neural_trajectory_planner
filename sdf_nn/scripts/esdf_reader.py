#!/usr/bin/env python

import rospy
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2

def callback(data):
    xmin = 100000
    xmax = -100000
    ymin = 100000
    ymax = -100000
    zmin = 100000
    zmax = -100000
    inmin = 100000
    inmax = -100000
    rospy.loginfo("Received a new PointCloud2 message")
    # Iterate through each point in the point cloud
    for p in pc2.read_points(data, field_names=("x", "y", "z", "intensity"), skip_nans=True):
        #rospy.loginfo("x: {}, y: {}, z: {}, intensity: {}".format(p[0], p[1], p[2], p[3]))
        if p[0] < xmin:
            xmin = p[0]
        if p[0] > xmax:
            xmax = p[0]
        if p[1] < ymin:
            ymin = p[1]
        if p[1] > ymax:
            ymax = p[1]
        if p[2] < zmin:
            zmin = p[2]
        if p[2] > zmax:
            zmax = p[2]
        if p[3] < inmin:
            inmin = p[3]
        if p[3] > inmax:
            inmax = p[3]
    print("Xmin = ", xmin, "  ||  Xmax = ", xmax)
    print("Ymin = ", ymin, "  ||  Ymax = ", ymax)
    print("Zmin = ", zmin, "  ||  Zmax = ", zmax)
    print("SDFmin = ", inmin, "  ||  SDFmax = ", inmax)


def listener():
    rospy.init_node('point_cloud_listener', anonymous=True)
    rospy.Subscriber("/data_slice", PointCloud2, callback)
    rospy.spin()

if __name__ == '__main__':
    listener()