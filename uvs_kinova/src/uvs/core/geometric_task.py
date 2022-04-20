#!/usr/bin/python3

"""
Geometric task for visuo-motor control.

"""

from abc import abstractmethod

import numpy as np
import rospy
from rospy.numpy_msg import numpy_msg

from uvs_kinova.msg import VisualTask
from uvs_kinova.msg import TrackerStates

from ..kg import KnowledgeGraph

# Base Class
# #####
class GeometricTask:
    """Base abstract class for a visuo-motor, geometric task.
    """
    def __init__(
        self,
        task_name: str,
        n_eef: int,
        n_goal: int,
        ):
        self.task_name = task_name
        self.n_eef = n_eef
        self.n_goal = n_goal

        # Store visual errors
        self.eef_specs, self.goal_specs = {}, {}

        # Publisher for visual error
        self.task_pub = rospy.Publisher('/uvs/visual_task', numpy_msg(VisualTask), queue_size=10)

    def __str__(self):
        return self.task_name

    def __call__(
        self, 
        inputs: dict,
        ):
        """Calculate and publish visual eef and goal to ROS.
        """
        eef, goal = self.get_eef_and_goal(inputs)

        # Publish eef and goal to ROS
        image_pose_msg = VisualTask()
        image_pose_msg.eef = eef
        image_pose_msg.goal = goal
        image_pose_msg.name = self.task_name
        self.task_pub.publish(image_pose_msg)

        return inputs

    @abstractmethod
    def get_eef_and_goal(
        self,
        inputs: dict,
        ):
        """Define how visual error to be calculated for the task.
        """
        raise NotImplementedError

    @abstractmethod
    def convert_to_kg(self):
        """Represent the geometric task as a knowledge graph.
        """
        raise NotImplementedError


# Individual Tasks
# #####

class PointToPointTask(GeometricTask):
    """One basic point-to-point task.
    """
    def __init__(
        self,
        task_name: str = "point_to_point",
        n_eef: int = 1,
        n_goal: int = 1,
        ):
        super().__init__(task_name, n_eef, n_goal)
    
    def get_eef_and_goal(
        self, 
        inputs: dict,
        ):
        """Publish point-to-point visual error.
        """
        eef_trackers, goal_trackers = [], []
        for i in range(1, self.n_eef + 1):
            eef_str, eef_corner = inputs["eef_{}".format(i)]
            eef_trackers.append(eef_corner)
            eef_str = "_".join(eef_str.split("_")[2:])
            self.eef_specs["eef_{}".format(i)] = [eef_str, eef_corner]

        for i in range(1, self.n_goal + 1):
            goal_str, goal_corner = inputs["goal_{}".format(i)]
            goal_trackers.append(goal_corner)
            goal_str = "_".join(goal_str.split("_")[2:])
            self.goal_specs["goal_{}".format(i)] = [goal_str, goal_corner]

        # Use double-loop to deal with m by n case
        # Can have 2 eef position, 1 goal
        eefs, goals = [], []
        for eef_corner in eef_trackers:
            # Collapse into a single CoM
            # TODO: Add grid behavior
            # TODO: Add depth frame behavior
            # TODO: Normalize coordinates
            eef = eef_corner.mean(axis=1)

            for goal_corner in goal_trackers:
                # TODO: Add depth frame behavior
                # TODO: Normalize coordinates
                goal = goal_corner.mean(axis=1)

                # Collect one correspondence
                eefs.append(eef)
                goals.append(goal)

        # Concatenate into one-dim arrays
        eef = np.concatenate(eefs, axis=0)
        goal = np.concatenate(goals, axis=0)

        return eef, goal

    def convert_to_kg(
        self,
        add_coords: bool = False,
        ):
        """Represent point-to-point task as a knowledge graph.
        A geometric task should specify the task intention.
        This serves as the main body, allowing dynamic leaves to be padded later. 
        Args:
            add_coords: Whether to add visual coordinates to the model node, 
                instance of eef or goal node.
        """
        intention_kg = KnowledgeGraph()
        print(self.eef_specs, self.goal_specs)
        intention_kg.add("EEFPoint", "uvs_move", "GoalPoint")

        # Construct logical instances for eef
        for _, eef_spec in self.eef_specs.items():
            model_node = "{}\n{}".format(eef_spec[0], eef_spec[1][:, 0]) if add_coords else eef_spec[0]
            intention_kg.add("EEFPoint", "hasInstance", model_node)
            intention_kg.add(model_node, "isInstanceOf", "EEFPoint")

        # Construct logical instances for goal
        for _, goal_spec in self.goal_specs.items():
            model_node = "{}\n{}".format(goal_spec[0], goal_spec[1][:, 0]) if add_coords else goal_spec[0]
            intention_kg.add("GoalPoint", "hasInstance", model_node)
            intention_kg.add(model_node, "isInstanceOf", "GoalPoint")

        return intention_kg