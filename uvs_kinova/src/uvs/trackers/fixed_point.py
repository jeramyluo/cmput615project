#!/usr/bin/python3

"""
Wrapper for pyMTF.

"""

import numpy as np
import cv2

from . import BaseTracker

class FixedPoint(BaseTracker):
    """Store a static, fixed point.
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
        self.point = corner_init

    def __str__(self):
        return "{}_fixed_point".format(self.name)

    def track(
        self, 
        frame: np.ndarray,
        ):
        """Only return the fixed point.
        """
        return self.point