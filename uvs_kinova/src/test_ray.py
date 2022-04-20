import glob 

import cv2 
import numpy as np 

from uvs.sce import estimate_ray_with_focal 
from utils import Tracker, draw_points, draw_arrows, get_angles 

FOCAL = 710 
IN_DEGREE = True 
VID_PATH = "../data/test_vid/test.mp4"
OUT_PATH = "../data/test_vid/test_out.mp4" 
WINDOW_NAME = "Test"

if __name__ == "__main__": 
    # read image video 
    vid = cv2.VideoCapture(VID_PATH) 

    # initialize tracker(s) 
    tracker = Tracker() 
    success, init_img = vid.read()  
    size = (int(init_img.shape[1] * 0.5), int(init_img.shape[0] * 0.5))
    out = cv2.VideoWriter(OUT_PATH, -1, 20.0, size)
    init_img = cv2.resize(init_img, size, interpolation=cv2.INTER_AREA) 
    img_center = ((init_img.shape[1]-1) // 2 + 1, (init_img.shape[0]-1) // 2 + 1) 
    img_center = np.array(img_center, dtype=float) 
    tracker.register_points(init_img, WINDOW_NAME)

    # main loop 
    while success: 
        # read image and update tracker 
        success, img = vid.read() 
        if not success: 
            break 
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
        tracker.update_tracker(gray) 

        # compute ray and angles from updated trackers 
        ray = estimate_ray_with_focal(img_center, FOCAL, tracker.points)[0]
        x_ang, y_ang = get_angles(ray, IN_DEGREE) 
        
        # display all the information 
        draw_points(img, tracker.points) 
        draw_points(img, np.expand_dims(img_center, axis=0))
        draw_arrows(img, img_center, img_center + ray[:2])       
        cv2.putText(img, "x angle: " + str(x_ang) + f"{'o' if IN_DEGREE else 'rad'}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(img, "z angle: " + str(y_ang) + f"{'o' if IN_DEGREE else 'rad'}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(img, "x vector: " + str(ray[0]) + " pix", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(img, "y vector: " + str(-ray[1]) + " pix", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(img, "z vector: " + str(ray[2]) + " pix", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)
        
        # save image 
        out.write(img)  
        cv2.imshow(WINDOW_NAME, img) 

        cv2.waitKey(1) 

    vid.release()
    out.release() 
    cv2.destroyAllWindows()
