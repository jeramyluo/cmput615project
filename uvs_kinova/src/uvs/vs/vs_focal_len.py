import numpy as np
import cv2
import time

from uvs.devices import KinovaGen3, RGBDVision
from uvs.sce import estimate_ray_with_focal
from utils import get_angles 

def get_central_diff(gen3, thetas, tracker, gray, img_center, cam):
    angles_to_send = [thetas[0], thetas[1], -179.4915313720703, thetas[2], 0.06742369383573531, 0, 89.88030242919922]
    gen3.send_joint_angles(np.asarray(angles_to_send))
    time.sleep(3)
    img = cam.frame 
    gray = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
    tracker.update_tracker(gray) 
    depth_map = cam.depth
    ray = (estimate_ray_with_focal(img_center, 640, tracker.points))[0][:2]
    # img: 640 x 480 
    # depth: 480 x 270
    x = int(tracker.points[0][0] * 270 / 480) + 60
    y = int(tracker.points[0][1] * 270 / 480)
    ray = np.append(ray, depth_map[y][x])
    return ray

def get_central_diff_angles(gen3, thetas, tracker, gray, img_center, cam):
    angles_to_send = [thetas[0], thetas[1], -179.4915313720703, thetas[2], 0.06742369383573531, 0, 89.88030242919922]
    gen3.send_joint_angles(np.asarray(angles_to_send))
    time.sleep(2)
    img = cam.frame
    gray = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
    tracker.update_tracker(gray)
    depth_map = cam.depth
    # img: 640 x 480 
    # depth: 480 x 270
    x = int(tracker.points[0][0] * 270 / 480) + 60
    y = int(tracker.points[0][1] * 270 / 480)
    # theta -> rotation in dir of cam
    # phi -> rotation in x dir
    theta, phi = get_angles((estimate_ray_with_focal(img_center, 640, tracker.points)))
    return np.array([theta, phi, depth_map[y][x]])

def get_pos_simple(img_center, FOCAL, current_goal, cam):
    current_ray = (estimate_ray_with_focal(img_center, FOCAL, current_goal)[0])[:2]
    depth_map = cam.depth
    x = int(current_goal[0][0] * 270 / 480) + 60
    y = int(current_goal[0][1] * 270 / 480) 
    current_ray = np.append(current_ray, depth_map[y][x])
    return current_ray

def get_angles_simple(img_center, FOCAL, current_goal, cam):
    depth_map = cam.depth
    x = int(current_goal[0][0] * 270 / 480) + 60
    y = int(current_goal[0][1] * 270 / 480)
    theta, phi = get_angles((estimate_ray_with_focal(img_center, FOCAL, current_goal)))
    return np.array([theta, phi, depth_map[y][x]])


def get_jacobian_simple(gen3, tracker, gray, img_center, cam):
    print("conducting J est\n")
    jacobian = []
    h = 1 # step size for central difference
    current_angles = (gen3.position)*180/np.pi
    to_send_og = current_angles
    thetas = np.array([current_angles[0], current_angles[1], current_angles[3]])

    for i in range(3):
        ray = []
        for j in range(2):
            thetas[i] = thetas[i] + (-1)**(j) * h
            ray.append(get_central_diff(gen3, thetas, tracker, gray, img_center, cam))
            gen3.send_joint_angles(to_send_og)
            time.sleep(1)
            thetas[i] = thetas[i] - (-1)**(j) * h
        jacobian.append((ray[0] - ray[1])/(2*h))
    print("finished J est\n")
    return np.asarray(jacobian).T

def get_jacobian_angles(gen3, tracker, gray, img_center, cam) :
    print("conducting J est")
    jacobian = []
    h = 1
    current_angles = (gen3.position)*180/np.pi
    to_send_og = current_angles
    thetas = np.array([current_angles[0], current_angles[1], current_angles[3]])

    for i in range(3):
        ray = []
        for j in range(2):
            thetas[i] = thetas[i] + (-1)**(j) * h
            ray.append(get_central_diff_angles(gen3, thetas, tracker, gray, img_center, cam))
            gen3.send_joint_angles(to_send_og)
            time.sleep(1)
            thetas[i] = thetas[i] - (-1)**(j) * h
        jacobian.append((ray[0] - ray[1])/(2*h))
    print("finished J est\n")
    return np.asarray(jacobian).T

def get_waypoint(pos, step):
    pt_wrt_center = []
    for i in range(3):
        if np.abs(pos[i]) < step:
            pt_wrt_center.append(pos[i])
        else:
            pt_wrt_center.append(np.sign(pos[i]) * step)
    pt_wrt_goal = pos - pt_wrt_center
    return pt_wrt_goal

    

    
