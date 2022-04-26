import numpy as np
import time
import math

from uvs.sce import estimate_ray_with_focal

def get_central_diff(gen3, thetas, cam, img_center):
    angles_to_send = [thetas[0], thetas[1], -179.4915313720703, thetas[2], 0.06742369383573531, 0, 89.88030242919922]
    gen3.send_joint_angles(np.asarray(angles_to_send))
    time.sleep(2)
    depth_map = cam.depth
    ray = (estimate_ray_with_focal(img_center, 640, cam.tracker.points))[0][:2]
    # img: 640 x 480 
    # depth: 480 x 270
    x = int(cam.tracker.points[0][0] * 270 / 480) + 60
    y = int(cam.tracker.points[0][1] * 270 / 480)
    ray = np.append(ray, depth_map[y][x])
    return ray


def get_pos_simple(img_center, FOCAL, current_goal, cam):
    current_ray = (estimate_ray_with_focal(img_center, FOCAL, current_goal)[0])[:2]
    depth_map = cam.depth
    x = int(current_goal[0][0] * 270 / 480) + 60
    y = int(current_goal[0][1] * 270 / 480) 
    current_ray = np.append(current_ray, depth_map[y][x])
    return current_ray


def get_jacobian_simple(gen3, cam, img_center):
    print("conducting J est\n")
    jacobian = []
    h = 2 # step size for central difference
    current_angles = (gen3.position)*180/np.pi
    to_send_og = current_angles
    thetas = np.array([current_angles[0], current_angles[1], current_angles[3]])

    for i in range(3):
        ray = []
        for j in range(2):
            thetas[i] = thetas[i] + (-1)**(j) * h
            ray.append(get_central_diff(gen3, thetas, cam, img_center))
            gen3.send_joint_angles(to_send_og)
            time.sleep(1)
            thetas[i] = thetas[i] - (-1)**(j) * h
        jacobian.append((ray[0] - ray[1])/(2*h))
    print("finished J est\n")
    return np.asarray(jacobian).T


def step_function(norm, function):
    if function == 'linear':
        step = 0.2*norm - 25
    if function == 'sigmoid':
        z = -0.006*(norm - 500)
        step = 160 / (1 + math.exp(z))
    if function == 'constant':
        step = 30
    return step


def get_waypoint(pos, align=True, function='sigmoid'):
    pt_wrt_center = []
    step = step_function(np.linalg.norm(pos), function)
    if align == True and np.linalg.norm(pos[0] + pos[1]) > 10:
        for i in range(2):
            if np.abs(pos[i]) < step:
                pt_wrt_center.append(pos[i])
            else:
                pt_wrt_center.append(np.sign(pos[i]) * step)
        pt_wrt_center.append(0)
    else:
        for i in range(3):
            if np.abs(pos[i]) < step:
                pt_wrt_center.append(pos[i])
            else:
                pt_wrt_center.append(np.sign(pos[i]) * step)
    pt_wrt_goal = pos - pt_wrt_center
    return pt_wrt_goal


def visual_servo(gen3, img_center, FOCAL, cam, jacobian=[]):

    function = 'sigmoid'
    align = True
    distance_threshold = 200
    sleep_time = 2


    if jacobian == []:
        print("Estimating Jacobian")
        jacobian = get_jacobian_simple(gen3, cam, img_center)
        print(f"Jacobian = {jacobian}")
    current_pos = get_pos_simple(img_center, FOCAL, cam.tracker.points, cam)
    current_pos_wp = np.array([0, 0, 0])
    error_threshold = 1
    iterations = 0

    while np.linalg.norm(current_pos) > distance_threshold:      
        if np.linalg.norm((current_pos_wp)) < (error_threshold):
            current_pos = get_pos_simple(img_center, FOCAL, cam.tracker.points, cam)
            goal_wrt_target = get_waypoint(current_pos, align, function)
            error_threshold = np.linalg.norm(goal_wrt_target)/4
            current_pos_wp = current_pos - goal_wrt_target

        print(current_pos_wp)    
        goal = current_pos_wp

        # solving for new joint angles
        sk = np.linalg.lstsq(jacobian, -1 * goal, rcond=None)[0]
        current_angles = (gen3.position)*180/np.pi

        # sending joint angles to robot
        joints = np.array([current_angles[0], current_angles[1], current_angles[3]])
        joints += sk
        to_send = [joints[0], joints[1], current_angles[2], joints[2], current_angles[4], current_angles[5], current_angles[6]]
        success = gen3.send_joint_angles(to_send)
        time.sleep(sleep_time)

        # computing next point
        next_pos_wp = get_pos_simple(img_center, FOCAL, cam.tracker.points, cam) - goal_wrt_target

        # updating jacobian
        yk = next_pos_wp - current_pos_wp
        jacobian += np.array([(yk - jacobian @ sk)]).transpose() @ np.array([sk])/(np.array([sk]) @ np.array([sk]).T)
        current_pos_wp = next_pos_wp
        if current_pos_wp[2] < 0:
            current_pos_wp[2] = 0    
        
        iterations += 1

    return iterations

    
