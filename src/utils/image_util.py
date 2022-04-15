#TODO: Develop image utility functions 
import cv2
import numpy as np 

class Tracker(object): 
    """Holds methods and attributes related to tracking algorithm. 

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
        window_name: str 
            Name of the window to show the plot. 

        Returns
        ----------
        points: np.ndarray 
            Spatial 2D coordinates in image to track. 
        
        I/O: 
        ----------
        Take in the initial image or arbitrary size. 
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
        
        if len(img.shape) > 2: 
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
        self.init_tracker(img) 

    def init_tracker(self, img: np.ndarray): 
        """Initialize tracker with first frame. 
        
        Parameters: 
        ----------
        img: np.ndarray 
            First frame of video. 

        I/O 
        ----------
        Take in a single image in gray scale. 
        """
        # globalize variables 
        self.prev_frame = img.copy()
        
    def update_tracker(self, img: np.ndarray): 
        """Update the trackers.
        
        Parameters: 
        ----------
        img: np.ndarray 
            Frame for updated view. 
        
        I/O
        ----------
        Take in a single gray scale image, which is in the same shape as in self.prev_frame.
        """
        # check if image is in gray scale or not 
        if len(img.shape) > 2: 
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 

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
        Arbitrary sized image to draw on.
    points: np.ndarray 
        Pixel coordinates to draw circles. 
    color: tuple 
        Color code of the points, represented as (B, G, R) where each elements are the in the integer between 0 ~ 255. 
    radius: int
        Radius of each circle in pixel scale. 
    thickness: int 
        Value of integer which specifies thickness of circles. 

    I/O
    ----------
    Takes in parameters above, then return nothing. 
    """
    if (points.shape[-1] != 2): 
        raise ValueError("Coordinates of tracking points must be in the shape of (n, 2), where n is the number of points.")
    
    # plot circle at all the point coordinates 
    for i in range(points.shape[0]): 
        x, y = int(points[i, 0]), int(points[i, 1])
        cv2.circle(img, (x, y), radius=radius, color=color, thickness=thickness)

def draw_arrows(img: np.ndarray, start_pts: np.ndarray, end_pts: np.ndarray, color: tuple=(0, 0, 255), thickness: int=5): 
    """Draw arrows on the image.
    
    Parameters: 
    ----------
    img: np.ndarray 
        Arbitrary sized image to draw on. 
    start_pts: np.ndarray 
        Start coordinates of arrows, where each pixel scale coordinates are in tuple (x, y). 
    end_pts: np.ndarray 
        End (tips) coordinates of arrows, where each pixel scale coordinates are in tuple (x, y) and each corresponds to start_pts.
    color: tuple 
        Color code of the points, represented as (B, G, R) where each elements are the in the integer between 0 ~ 255. 
    thickness: int 
        Value of integer which specifies thickness of arrows. 
    """ 
    if (start_pts.shape[-1] != 2 or start_pts.shape[-1] != 2): 
        raise ValueError("Coordinates of starting/ending points must be in the shape of (n, 2), where n is the number of arrows.")
    if (start_pts.shape != end_pts.shape): 
        raise ValueError("Starting points and end points must have the exactly same shape. Instead got {0} and {1}".format(start_pts.shape, end_pts.shape))
    
    # avoid dimension error by checking the shape (i.e. avoid shape of (2, )) 
    if len(start_pts.shape) == 1: 
        start_pts = np.expand_dims(start_pts, axis=0) 
    if len(end_pts.shape) == 1: 
        end_pts = np.expand_dims(end_pts, axis=0) 

    for i in range(start_pts.shape[0]): 
        cv2.arrowedLine(img, start_pts[i, :].astype(int), end_pts[i, :].astype(int), color=color, thickness=thickness)
    
def get_angles(ray: np.ndarray, to_degree: bool = True): 
    """Compute angles in both x, y directions with respect to principal axis (i.e. axis perpendicular to image plane). 

    Parameters: 
    ----------
    ray: np.ndarray 
        Direction vector of ray from center of image, with shape (1x3) or (3, ). 
        Vector must be in (x, y, z) where: 
            x: horizontal direction on image plane. 
            y: vertical direction on image plane. 
            z: principal axis.
    to_degree: bool 
        Decide whether to convert from radian to degree. 

    Returns: 
    ----------
    x_ang: float
        Angle in xy direction, equivalent to rotation of ray on image plane.  
    z_ang: float 
        Angle in z direciton, equivalent rotation of ray from principal axis.  

    I/O
    -----------
    Takes in (3, 1), (1x3) or (3,) shaped vector then return two floating point numbers as specified in Returns section. 
    """
    x, y, z = ray 
    xy_norm = np.linalg.norm(ray[:2])
    x_ang = np.arccos(x/xy_norm)
    x_ang *= -1 if y > 0 else 1 
    z_ang = np.pi/2 - np.arctan(z/xy_norm) 

    if to_degree: 
        x_ang *= 180 / np.math.pi 
        z_ang *= 180 / np.math.pi 
    
    return x_ang, z_ang 

