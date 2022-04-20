#!/usr/bin/python3

"""
Template object for a visual tracker.

"""

from abc import abstractmethod
import numpy as np
import cv2
import rospy

from uvs_kinova.msg import TrackerStates

class BaseTracker(object):
    """Warp a simple tracker from MTF.
    """
    def __init__(
        self,
        name: str,
        frame_init: np.ndarray,
        corner_init: np.ndarray,
        cfg,
        visualize: bool = False,
        ):
        self.name = name
        self.cfg = cfg
        self.visualize = visualize

        self.tracker_pub = rospy.Publisher(
            "/uvs/{}".format(str(self)), 
            TrackerStates, 
            queue_size=10)

        # Small distinguish on color
        if 'eef' in self.name:
            self.visualize_color = (0, 255, 0)
        else:
            self.visualize_color = (0, 0, 255)

    @abstractmethod
    def __str__(self):
        raise NotImplementedError
    
    @abstractmethod
    def track(
        self, 
        frame: np.ndarray,
        ):
        """Method to update the tracker.
        """
        raise NotImplementedError

    def __call__(
        self, 
        inputs: dict,
        ):
        """Determine behavior of a tracker / model in general.
        """
        # Track
        corner = self.track(inputs['frame'])
        inputs[self.name] = [str(self), corner]

        # Visualize
        if self.visualize:
            if corner.shape[1] > 1:
                cv2.polylines(inputs['vis'], [corner.T], True, self.visualize_color, 2)
            else:
                cv2.circle(inputs['vis'], (corner[0, 0], corner[1, 0]), 2, self.visualize_color, 2)

        # Publish tracking results to ROS
        state_msg = TrackerStates()
        state_msg.name = str(self)
        state_msg.x = corner[0, :]
        state_msg.y = corner[1, :]
        self.tracker_pub.publish(state_msg)

        return inputs