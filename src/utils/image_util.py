#TODO: Develop image utility functions 
import cv2
import numpy as np 
from cv_bridge import CvBridgeError

from uvs.kinova_controller import RGBDCamera 

class CapturedCam(RGBDCamera): 
    def __init__(self):
        super().__init__()

    def _image_callback(self, ros_image):
        try: 
            frame = self.bridge.imgmsg_to_cv2(ros_image, "mono8")
        except CvBridgeError as e: 
            print(e)
        self.frame = np.array(frame, dtype=np.uint8)

class Tracker(object): 
    """Holds methods or attributes related to tracking algorithm. 

    Attributes: 
    ----------
    points: np.ndarray 
        Coordinates of trackers. 
    prev_frame: np.ndarray 
        Previous frame of tracker.  
    """
    def __init__(self): 
        # for testing purpose 
        self.points = None
        self.prev_frame = None

    def register_points(self, img: np.ndarray, window_name: str=None): 
        """Register the points to track.

        Parameters 
        ---------- 
        img: np.ndarray 
            Image to use for registering point(s) to track.

        Returns
        ----------
        points: np.ndarray 
            Spatial 2D coordinates in image to track. 
        """
        global res 
        res = []
        
        def click_event(event, x, y, flags, params): 
            global res 
            # store and show the clicked coordinates when left click on mouse happens 
            if event == cv2.EVENT_LBUTTONDOWN: 
                res.append([x, y]) 
            cv2.circle(img, (x, y), 3, (0, 255, 0), 2)

        # initialize opencv window
        if window_name is None: 
            window_name = "Register" 

        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, click_event)
        
        # run until 'q' key is pressed 
        while 1: 
            cv2.imshow(window_name, img)
            if cv2.waitKey(1) & 0xFF == ord('q'): 
                break 

        # return coordinates in numpy array format 
        points = res.copy() 
        points = np.asarray(points, dtype=np.float32) 
        cv2.destroyAllWindows()
        self.points = points 

    def init_tracker(self, img: np.ndarray): 
        """Initialize tracker with first frame. 
        
        Parameters: 
        ----------
        img: np.ndarray 
            First frame of video. 
        """
        # globalize variables 
        self.prev_frame = img.copy()
        
    def update_tracker(self, img: np.ndarray): 
        """Update the trackers.
        
        Parameters: 
        ----------
        img: np.ndarray 
            Frame for updated view. 
        """
        lk_params = dict(winSize=(32, 32), 
                        maxLevel=8, 
                        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 9, 0.02555))
        p1, _, _ = cv2.calcOpticalFlowPyrLK(self.prev_frame, img, self.points, None, **lk_params)
        self.prev_frame = img.copy()
        self.points = p1.copy()

def draw_points(img: np.ndarray, points: np.ndarray, color: tuple=(0, 255, 0), radius: int=2, thickness: int=1): 
    """Draw points on the image. 

    Parameters: 
    ----------
    img: np.ndarray 
        Image to draw on.
    points: np.ndarray 
        Coordinates of circles. 
    color: tuple 
        Color code of the arrows. 
    radius: int
        Radius of each circle. 
    thickness: int 
        Value of integer which specifies thickness of arrows. 
    """
    if (points.shape[-1] != 2): 
        raise ValueError("Coordinates of tracking points must be in the shape of (n, 2), where n is the number of points.")
    
    # plot circle at all the point coordinates 
    for i in range(points.shape[0]): 
        x, y = int(points[i, 0]), int(points[i, 1])
        cv2.circle(img, (x, y), radius=radius, color=color, thickness=thickness)

def draw_arrows(img: np.ndarray, start_pts: np.ndarry, end_pts: np.ndarray, color: tuple=(0, 0, 255), thickness: int=5): 
    """Draw arrows on the image.
    
    Parameters: 
    ----------
    img: np.ndarray 
        Image to draw on.
    start_pts: np.ndarray 
        Start coordinates of arrows. 
    end_pts: np.ndarray 
        End (tips) coordinates of arrows. 
    color: tuple 
        Color code of the arrows. 
    thickness: int 
        Value of integer which specifies thickness of arrows. 
    """ 
    if (start_pts.shape[-1] != 2 or start_pts.shape[-1] != 2): 
        raise ValueError("Coordinates of starting/ending points must be in the shape of (n, 2), where n is the number of arrows.")
    if (start_pts.shape != end_pts.shape): 
        raise ValueError("Starting points and end points must have the exactly same shape. Instead got {0} and {1}".format(start_pts.shape, end_pts.shape))
    
    for i in range(start_pts.shape[0]): 
        cv2.arrowedLine(img, start_pts[i, :], end_pts[i, :], color=color, thickness=thickness)
    
