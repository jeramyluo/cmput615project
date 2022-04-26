#!/usr/bin/python3

"""
Uncalibrated Visual Servoing.

"""

import numpy as np
import rospy
from rospy.numpy_msg import numpy_msg

from uvs_kinova.msg import VisualTask


class Servo:
    """Main control class for UVS.
    """
    def __init__(
        self, 
        robot: object,
        active_joints: list,
        camera_type: str = "eye_in_hand",
        ):
        self.robot = robot
        self.active_joints = active_joints
        self.camera_type = camera_type

        # Subscriber
        try:
            # Setup visual error subscriber
            self.eef, self.goal = None, None
            self.task_name = None
            self.task_sub = rospy.Subscriber('/uvs/visual_task', numpy_msg(VisualTask), self._task_callback)

        except:
            raise Exception("Failed to communicate w/ ROS.")

        # Numeric
        self._wait_for_task()
        self.jacobian_initialized = False
        self.J = np.zeros((len(self.eef), len(self.active_joints)), dtype=np.float64)   # Image Jacobian

    def _wait_for_task(self):
        """Wait until receving a reading of task.
        """
        while self.eef is None or self.goal is None:
            rospy.sleep(0.001)

    def _task_callback(
        self,
        data,
        ):
        self.eef = data.eef
        self.goal = data.goal
        self.task_name = data.name
        #print(self.eef, self.goal, self.task_name)

    def ortho_exploratory_motions(
        self,
        delta: float = 8.5):
        """Estimate initial image jacobian w/ 
        orthogonal exploratory motion.
        """
        current_angles = self.robot.position
        print(current_angles)

        for active_joint in self.active_joints:
            delta_angles = np.zeros((current_angles.shape[0], ), dtype=current_angles.dtype)
            delta_angles[self.robot.JOINT_NAME_TO_ID[active_joint]] = delta

            # Plus
            self.robot.send_joint_angles(current_angles + delta_angles)
            rospy.sleep(1.0)
            # TODO: Get change in image coordinates
            # #####
            
            # #####
            

            # Minuss
            self.robot.send_joint_angles(current_angles - delta_angles)
            rospy.sleep(1.0)
            # TODO: Get change in image coordinates
            # #####
            
            # #####

            #self.J = None

            # Return back to original position
            self.robot.send_joint_angles(current_angles)
            rospy.sleep(1.0)

        self.jacobian_initialized = True

    