#!/usr/bin/python3
# TODO: figure out stiff joint angles before testing
import numpy as np
import cv2
import rospy
import time

from uvs.devices import KinovaGen3, RGBDVision
from uvs.sce import estimate_ray_with_focal 
from uvs.vs import get_pos_simple, get_jacobian_simple, get_jacobian_angles, get_angles_simple, get_waypoint
from utils import Tracker, draw_points, draw_arrows, get_angles
from uvs.vs.vs_focal_len import get_waypoint 

if __name__ == "__main__":
    # Init ros node
    rospy.init_node('gen3_servo', anonymous=False)

    # Gen3 camera node
    gen3 = KinovaGen3(init_camera=True)
    cam = RGBDVision('my_gen3')

    FOCAL = 640
    IN_DEGREE = True

    # Send default pos to robot
    angles = np.array([0.1336059570312672, -25, -179.4915313720703, -140, 0.06742369383573531, 0, 89.88030242919922])
    success = gen3.send_joint_angles(angles)
    time.sleep(5)
    print("Done. Joints:", gen3.position)

    # init tracker here
    tracker = Tracker() 
    init_img = cam.frame 
    img_center = ((init_img.shape[1]-1) // 2 + 1, (init_img.shape[0]-1) // 2 + 1) 
    img_center = np.array(img_center, dtype=float) 
    tracker.register_points(init_img, 'test')
    cam.set_tracker(tracker)

    img = cam.frame 
    gray = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
    tracker.update_tracker(gray)
    iterations = 0


    # jacobian = get_jacobian_angles(gen3, tracker, gray, img_center, cam)
    # current_angle = get_angles_simple(img_center, FOCAL, tracker.points, cam)/10
    # while iterations < 10:
    #     print(jacobian)
    #     print(current_angle)
    #     # sk = (np.linalg.lstsq(jacobian, -1 * current_pos, rcond=None))[0]
    #     sk = np.linalg.solve(jacobian, -1 * current_angle)
    #     current_joints = (gen3.position)*180/np.pi
    #     next_joints = np.array([current_joints[0], current_joints[1], current_joints[3]])
    #     next_joints += sk
    #     to_send = [next_joints[0], next_joints[1], current_joints[2], next_joints[2], current_joints[4], current_joints[5], current_joints[6]]
    #     print(to_send)
    #     success = gen3.send_joint_angles(to_send)
    #     time.sleep(1)
    #     img = cam.frame 
    #     gray = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
    #     tracker.update_tracker(gray)
    #     next_angle = get_angles_simple(img_center, FOCAL, tracker.points, cam)/10
    #     yk = next_angle - current_angle
    #     jacobian += np.array([(yk - jacobian @ sk)]).transpose() @ np.array([sk])/(np.array([sk]) @ np.array([sk]).T)
    #     current_angle = next_angle
    #     iterations += 1

    jacobian = get_jacobian_simple(gen3, tracker, gray, img_center, cam)
    # jacobian = [[-15.5   1.    0. ] [ -1.  -13.5  20.5][  1.5  -4.   -0.5]]
    current_pos_wp = np.array([0, 0, 0])
    waypoint_step = 10

    while iterations < 50:

        # get new waypoint when close to waypoint goal
        if np.linalg.norm((current_pos_wp)) < 5:
            current_pos = get_pos_simple(img_center, FOCAL, tracker.points, cam)
            goal_wrt_target = get_waypoint(current_pos, waypoint_step)
            current_pos_wp = current_pos - goal_wrt_target
        
        #set goal as current wp
        goal = current_pos_wp

        # printing test parameter
        print(jacobian)
        print(current_pos_wp)
        print(cam.tracker.points)

        # solving for new joint angles
        sk = np.linalg.lstsq(jacobian, -1 * goal, rcond=None)[0]
        current_angles = (gen3.position)*180/np.pi

        # sending joint angles to robot
        joints = np.array([current_angles[0], current_angles[1], current_angles[3]])
        joints += sk
        to_send = [joints[0], joints[1], current_angles[2], joints[2], current_angles[4], current_angles[5], current_angles[6]]
        print(to_send)
        success = gen3.send_joint_angles(to_send)
        time.sleep(1)

        # updating tracker
        # img = cam.frame
        # gray = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
        # tracker.update_tracker(gray)

        # computing next point
        next_pos_wp = get_pos_simple(img_center, FOCAL, cam.tracker.points, cam) - goal_wrt_target

        # updating jacobian
        yk = next_pos_wp - current_pos_wp
        jacobian += np.array([(yk - jacobian @ sk)]).transpose() @ np.array([sk])/(np.array([sk]) @ np.array([sk]).T)
        current_pos_wp = next_pos_wp
        iterations += 1

    #     # # compute ray 
    #     # ray = estimate_ray_with_focal(img_center, FOCAL, tracker.points)

    #     # # display all the information 
    #     draw_points(img, tracker.points) 
    #     cv2.imshow("Test", img)
    #     cv2.waitKey(1)
    #     # draw_points(img, np.expand_dims(img_center, axis=0))
    #     # draw_arrows(img, np.expand_dims(img_center, axis=0), np.expand_dims(img_center + ray[0, :2], axis=0))       
    #     #cv2.putText(img, "x angle: " + str(x_ang) + f"{'o' if IN_DEGREE else 'rad'}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)
    #     #cv2.putText(img, "z angle: " + str(y_ang) + f"{'o' if IN_DEGREE else 'rad'}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)
    #     #cv2.putText(img, "x vector: " + str(ray[0]) + " pix", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)
    #     #cv2.putText(img, "y vector: " + str(-ray[1]) + " pix", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)
    #     #cv2.putText(img, "z vector: " + str(ray[2]) + " pix", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)

    #     # implement target_reached check?? if ray is small break

    # # try to servo along waypoints? -> split ray into sections

    cv2.destroyAllWindows()