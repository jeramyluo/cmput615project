import glob 

import cv2 

from image_util import Tracker, draw_points 

tracker = Tracker()
img_dir = "../../../labs/lab3/data/vid/"
img_paths = glob.glob(img_dir + "*.png")
img_paths.sort(key=len)
window_name = "Tracker test"
color = (0, 255, 0)
radius = 2 
thickness = 1

init_img = cv2.imread(img_paths[0], 0)
tracker.register_points(init_img, window_name=window_name)
tracker.init_tracker(init_img)

cv2.namedWindow(window_name)

for img_path in img_paths[1:]:
    img = cv2.imread(img_path, 0)
    tracker.update_tracker(img)
    pts = tracker.points 

    draw_points(img, pts, color=color, radius=radius, thickness=thickness)
    cv2.imshow(window_name, img)

    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break 

cv2.destroyAllWindows()
    
