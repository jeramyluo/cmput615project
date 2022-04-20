#!/usr/bin/python3

"""
Wrapper for pyMTF.

"""

import numpy as np
import cv2
import pyMTF

from . import BaseTracker

class MTFTracker(BaseTracker):
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
        super().__init__(name, frame_init, corner_init, cfg, visualize)
        self.name = name

        # Initialize MTF with 0th frame and (2, 4) corner
        self.tracker_id = pyMTF.create(
            cv2.cvtColor(frame_init, cv2.COLOR_BGR2RGB), 
            np.float64(corner_init), 
            self.cfg)

    def __str__(self):
        return "{}_mtf_tracker".format(self.name)

    def track(
        self, 
        frame: np.ndarray,
        ):
        """Wrap around MTF update method.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        corner = np.zeros((2, 4), dtype=np.float64)
        success = pyMTF.getRegion(rgb, corner, self.tracker_id)
        self.corner = corner

        return np.round(corner).astype(int)