#!/usr/bin/python3

import numpy as np
import cv2
import rospy

import os
from uvs.kinova_controller import KinovaGen3

if __name__ == "__main__":
    # Init ros node
    rospy.init_node('/gen3/uvs', anonymous=False)

    # Camera node
    robot = KinovaGen3()

    # Spin
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down...")
        cv2.DestroyAllWindows()