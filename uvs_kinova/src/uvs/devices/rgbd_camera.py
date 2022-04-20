#!/usr/bin/python3

"""
Communicate w/ RGB-D camera through ROS + Forward Inference with any Vision Model.

"""

from matplotlib.pyplot import draw
import numpy as np
import cv2
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from utils import draw_points

class RGBDVision(object):
    """Interact with any camera w/ topics published to ROS.
    Standard cv_bridge. 

    To run with eye-in-hand camera on Kinova Gen3, install ros_kortex_vision, then run:
        roslaunch kinova_vision kinova_vision.launch num_worker_threads:=0
    """
    def __init__(
        self,
        name: str,
        image_topic: str = "/camera/color/image_raw",
        depth_topic: str = "/camera/depth/image_raw",
        image_encoding: str = "bgr8",
        depth_encoding: str = "passthrough",
        ):
        self.name = name
        self.image_encoding = image_encoding
        self.depth_encoding = depth_encoding
        
        # Image frame & depth
        self.frame = None
        self.depth = None
        self.i = 0
        self.clicks = []
        self.enable_mouse_event = False 

        # initialize tracker 
        self.tracker = None 

        # Additional ops needed to run in align with image pipeline
        self.update_ops = {}

        # Subscriber
        try:
            self.bridge = CvBridge()
            self.image_sub = rospy.Subscriber(image_topic, Image, self._image_callback)
            self.depth_sub = rospy.Subscriber(depth_topic, Image, self._depth_callback)
        except:
            raise Exception("Failed to communicate with a camera!")

    def _image_callback(
        self,
        ros_image,
        ):
        """Handle raw image frame from Kinova Gen3 camera.
        BGR Frame: (480, 640, 3)
        """
        try:
            self.frame = self.bridge.imgmsg_to_cv2(ros_image, self.image_encoding)
        except CvBridgeError as e:
            print(e)
                       
        # Display
        vis = self.frame.copy()
        for (x, y) in self.clicks:
            cv2.circle(vis, (x, y), 2, (0, 0, 255), 2)

        # NOTE: Now run any additional update_ops
        data = {
            'vis': vis,
            'frame': self.frame,
        }
        try:
            for _, ops in self.update_ops.items():
                data = ops(data)
        except RuntimeError:    # Ignore badcallback situation
            pass

        # imshow
        if self.tracker is not None:
            gray = cv2.cvtColor(data['vis'], cv2.COLOR_BGR2RGB)
            self.tracker.update_tracker(gray)
            draw_points(data['vis'], self.tracker.points, color=(0, 0, 255))

        cv2.imshow("{} RGB".format(self.name), data['vis'])
        if self.depth is not None: cv2.imshow("{} Depth".format(self.name), np.uint8(self.depth))
        key = cv2.waitKey(1)

        # NOTE: Remove later, simple save script
        if key == ord('s'):
            if self.frame is not None:
                cv2.imwrite("frame_{}.png".format(self.i), self.frame)
            if self.depth is not None:
                np.save("depth_{}.npy".format(self.i), self.depth)
            self.i += 1
            print("Image saved.")

        elif key == ord('c') and self.enable_mouse_event:   # ASCII dec: c
            cv2.setMouseCallback("{} RGB".format(self.name), self._mouse_event)

    def _depth_callback(
        self,
        ros_image,
        ):
        """Handle raw depth image from Kinova Gen3 camera.
        depth: (270, 480), uint16
        """
        try:
            # ros_kortex_vision publishes depth frame as 16UC1 encpding
            self.depth = self.bridge.imgmsg_to_cv2(ros_image, self.depth_encoding)
        except CvBridgeError as e:
            print(e)

        # Some image processing
        # TODO: Correctly normalize depth image frame
        # #####
        # #####

    def set_tracker(self, tracker): 
        self.tracker = tracker 

    # #####
    # Hook to OpenCV GUI
    # #####

    def clear_clicks(self):
        """Clear all clicked points.
        """
        self.clicks = []
    
    def _mouse_event(self, event, x, y, flags, param):
        """OpenCV mouse event. 
        Create one tracker per 4 clicked pts / 1 corner.
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            self.clicks.append([x, y])
            cv2.setMouseCallback("{} RGB".format(self.name), lambda *args: None)
            #print("Clicked points:", self.clicks)


    # #####
    # models / update_ops to perform
    # #####

    def add_update_ops(
        self,
        ops,
        ):
        """Append an update_ops (tracker, model, ...) to 
        run along with image_callback.
        """
        self.update_ops[str(ops)] = ops

    def del_update_ops(
        self, 
        ops_id: int,
        ):
        """Delete an update_ops (tracker, model, ...).
        """
        if ops_id in self.update_ops:
            del self.update_ops[ops_id]

    