import cv2 
import numpy as np 

from sce import Odometry, estimate_ray_with_proj
from utils import Tracker, draw_points, draw_arrows, get_angles  

if __name__ == "__main__": 
    ##### For test purpose, we use the known intrinsic params ##### 
    FOCAL = 710 
    IN_DEGREE = True 
    VID_PATH = "../data/test_vid/test.mp4"
    OUT_PATH = "../data/test_vid/test_out_odom.mp4" 
    WINDOW_NAME = "Test"

    # read video and configure the image  
    vid = cv2.VideoCapture(VID_PATH) 

    success, img = vid.read()
    if not success: 
        raise Exception("Video not loaded successfully.") 
    size = (int(img.shape[1] * 0.5), int(img.shape[0] * 0.5)) 
    out = cv2.VideoWriter(OUT_PATH, -1, 20.0, size) 
    if VID_PATH != "../data/test_vid/cup_test.mp4": 
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA) 
    center = ((img.shape[1]-1)//2+1, (img.shape[0]-1)//2+1)
    center = np.array(center, dtype=float) 

    # initialize tracker 
    tracker = Tracker() 
    tracker.register_points(img, WINDOW_NAME) 

    # initialize odometer 
    odom = Odometry(img, FOCAL, center) 

    cnt = 0
    orig_coord = None 
    while success: 
        # read next frame 
        success, img = vid.read() 
        if not success: 
            break 
        if VID_PATH != "../data/test_vid/cup_test.mp4": 
            img = cv2.resize(img, size, interpolation=cv2.INTER_AREA) 

        # update tracker and odometry 
        tracker.update_tracker(img) 
        err = odom.compute_odometry(img) 

        # print(odom.n_features) 
        ray_3d, ray_pixel = estimate_ray_with_proj(odom.P, tracker.points, odom.focal_len)
        x_ang, y_ang = get_angles(ray_3d[:3], False) 
        draw_points(img, tracker.points) 
        draw_arrows(img, center, center + ray_pixel[:2])
        # draw_arrows(img, center, center + np.array([r*np.cos(x_ang), r*np.sin(x_ang)]))
        # cv2.putText(img, f"{err}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)
        # cv2.putText(img, f"{ray}", ())
        # cv2.putText(img, "x angle: " + str(x_ang), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)
        # cv2.putText(img, "z angle: " + str(y_ang), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)
        out.write(img) 
        cv2.imshow("WINDOW_NAME", img) 
        cv2.waitKey(1) 

    vid.release() 
    out.release() 
    cv2.destroyAllWindows() 
