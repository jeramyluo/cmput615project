#!/usr/bin/python3

"""
Wrapper for pyMTF.

"""

import numpy as np
import cv2

from . import BaseTracker

class CSRTTracker(BaseTracker):
    """Warp CSRT tracker from OpenCV.
    """
    def __init__(
        self,
        name: str,
        frame_init: np.ndarray,
        corner_init: np.ndarray,
        cfg,
        visualize: bool = False,
        ):
        super().__init__(name, frame_init, corner_init, cfg, visualize)
        self.name = name

        # Initialize OpenCV CSRT tracker
        self.csrt = cv2.TrackerCSRT_create()

        x, y = corner_init[0, 0], corner_init[1, 0]
        w, h = corner_init[0, 2] - x, corner_init[1, 2] - y
        _ = self.csrt.init(
            cv2.cvtColor(frame_init, cv2.COLOR_BGR2RGB), 
            [x, y, w, h])

    def __str__(self):
        return "{}_csrt_tracker".format(self.name)

    def track(
        self, 
        frame: np.ndarray,
        ):
        """Wrap around CSRT update method.
        """
        _, bbox = self.csrt.update(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        x, y, w, h = bbox
        corner = np.array([
            [x, x + w, x + w, x],
            [y, y, y + h, y + h],
            ])
        self.corner = corner
        return np.round(corner).astype(int)