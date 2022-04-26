#!/usr/bin/python3
# TODO: figure out stiff joint angles before testing
from tkinter import W
import numpy as np
import cv2
import rospy
import time

from uvs.devices import KinovaGen3, RGBDVision
from uvs.vs import visual_servo
from utils import Tracker

if __name__ == "__main__":
    # Init ros node
    rospy.init_node('gen3_servo', anonymous=False)

    # Gen3 camera node
    gen3 = KinovaGen3(init_camera=True)
    cam = RGBDVision('my_gen3')

    FOCAL = 638.88
    IN_DEGREE = True

    # Send default pos to robot
    gen3.send_gripper_command(0)
    angles = np.array([0.1336059570312672, -25, -179.4915313720703, -140, 0.06742369383573531, 0, 89.88030242919922])
    success = gen3.send_joint_angles(angles)
    
    input("Press Enter to start")
    jacobian = np.array([[-14.25, 0.75, 0.], [-0.75, -13.25, 18.], [2.5, -6.75, -3.]])
    cont = "y"
    while cont == "y":
        tracker = Tracker() 
        init_img = cam.frame
        img_center = ((init_img.shape[1]-1) // 2 + 1, (init_img.shape[0]-1) // 2 + 1) 
        img_center = np.array(img_center, dtype=float)
        tracker.register_points(init_img, 'test')
        cam.set_tracker(tracker)
        start = time.time()
        iter = visual_servo(gen3, img_center, FOCAL, cam, jacobian)
        elapsed = time.time() - start
        print(f"Success! Target reached in {iter} iterations in {elapsed}s.")
        grip = input("Enter g to grab object")
        if grip == 'g':
            gen3.send_gripper_command(1)
        angles = np.array([0.1336059570312672, -25, -179.4915313720703, -140, 0.06742369383573531, 0, 89.88030242919922])
        success = gen3.send_joint_angles(angles)
        gen3.send_gripper_command(0)
        cont = input("Press y to continue")

    cv2.destroyAllWindows()