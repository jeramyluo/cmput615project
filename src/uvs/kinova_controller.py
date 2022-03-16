#!/usr/bin/python3

import numpy as np
import cv2
import rospy

from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image

class RGBDCamera(object):
    """Interact with eye-in-hand camera on Kinova Gen3.
    Standard cv_bridge. Run along with ros_kortex_vision:
        roslaunch kinova_vision kinova_vision.launch
    """
    def __init__(
        self,
        ):
        self.bridge = CvBridge()
        self._subscribe()

        self.frame = None
        self.depth = None
        self.i = 0

    def _subscribe(self):
        self.image_sub = rospy.Subscriber("/camera/color/image_raw", Image, self._image_callback)
        self.depth_sub = rospy.Subscriber("/camera/depth/image_raw", Image, self._depth_callback)

    def _image_callback(
        self,
        ros_image,
        ):
        """Handle raw image frame from Kinova Gen3 camera.
        RGB Frame: (480, 640, 3)
        """
        try:
            frame = self.bridge.imgmsg_to_cv2(ros_image, "bgr8")
        except CvBridgeError as e:
            print(e)
        self.frame = np.array(frame, dtype=np.uint8)
                       
        # Display
        vis = frame.copy()
        cv2.imshow("Kinova RGB", vis)
        
        key = cv2.waitKey(3)
        if key == ord('q'):
            rospy.signal_shutdown("Q pressed. Exiting...")

        # NOTE: Remove later, simple save script
        elif key == ord('s'):
            if self.frame is not None:
                cv2.imwrite("frame_{}.png".format(self.i), self.frame)
            if self.depth is not None:
                np.save("depth_{}.npy".format(self.i), self.depth)
            self.i += 1
            print("Image saved.")

    def _depth_callback(
        self,
        ros_image,
        ):
        """Handle raw depth image from Kinova Gen3 camera.
        depth: (270, 480), uint16
        """
        try:
            # ros_kortex_vision publishes depth frame as 16UC1 encpding
            depth = self.bridge.imgmsg_to_cv2(ros_image, "16UC1")
        except CvBridgeError as e:
            print(e)
        self.depth = depth

        # Some image processing
        # TODO: Correctly normalize depth image frame
        # #####
        # #####

        # Display
        vis = np.uint8(depth.copy())
        cv2.imshow("Kinova Depth", vis)


class KinovaGen3(object):
    """Kinova Gen3.
    Interact with robot control.
    """
    def __init__(
        self,
        name: str = "KinovaGen3",
        degrees_of_freedom: int = 7,
        ):
        self.name = name
        self.degrees_of_freedom = degrees_of_freedom

        self.camera = RGBDCamera()

        # ####################
        # TODO:
        # Receiving robotic pose into python
        # ####################
        self.pose = None


    def reach(
        self,
        ):
        return

    def __str__(self):
        return "{}, DOF: {}".format(self.name, self.degrees_of_freedom)