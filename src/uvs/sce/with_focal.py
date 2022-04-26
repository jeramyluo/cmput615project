import numpy as np 

def compute_focal_length(x1: int, x2: int, z: float, x: float): 
    """Compute focal length using similar triangle property.
    
    Parameters
    ----------
    x1: int 
        First reference coordinate on 2D image. x coordinate of single point.
    x2: int 
        Second reference coordinate on 2D image. x coordinate of another point from the point corresponding to x1.
    z: int 
        Known depth from image plane to the object. Usually in centimeters.
    x: int 
        Known distance between x1 and x2 in 3D world. Usually in centimeters.
    
    Returns
    ----------
    focal_len: int 
        Focal length of the camera. 

    I/O
    ----------
    Takes in inputs as in Parameters, then return single integer which represents focal length. 

    Note
    ----------
    Measurement scale on inputs must be the same. For instance, If z is in centimeters, x must be in centimeters. 
    """
    # compute pixel scale focal length from given values 
    pixel_diff = abs(x1 - x2)
    focal_len = pixel_diff * z / x  

    return focal_len

def estimate_ray_with_focal(img_center: tuple, focal_len: int, tracker_points: np.ndarray): 
    """Compute direction vector with known focal lengths and tracker point.
    
    Parameters
    ----------
    img_center: tuple 
        Center coordinate of image in form of (x, y) where each values are in integer. Usually it is the half of width and height of image. 
    focal_len: int 
        Known focal length in pixel scale. 
    tracker_points: np.ndarray
        Pixel coordinates of tracker point(s). It is ndarray with shape of (n, 2) where n is the number of tracking points.
    
    Returns
    ----------
    pixel_rays: np.ndarray 
        Estimated rays in pixel scale. 
    
    I/O
    ----------
    Takes in all the parameters as specified above, and returns pixel scale ray in format of (x, y, z). 
    """
    assert focal_len > 0, "Focal length must be int greater than 0." 
    assert tracker_points.shape[-1] == 2, "Tracker coordinates must be in the format of (n, 2) or (2, )."

    if isinstance(img_center, tuple) or isinstance(img_center, list): 
        img_center = np.array(img_center)
    
    diff = (tracker_points - img_center).astype(np.int32) 
    pixel_rays = list()
    for i in range(len(diff)):   
        # store ray vectors in [x, y, z] coordinate form, assuming z is the axis along focal length 
        pix_ray = [diff[i, 0], diff[i, 1], focal_len]

        # store all the results 
        pixel_rays.append(pix_ray)
    
    return np.asarray(pixel_rays)
