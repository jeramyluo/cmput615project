import numpy as np 

from sce import get_intrinsics_from_checker 

if __name__ == "__main__": 
    img_dir = "../data/chess/" 
    mat, dist, rvecs, tvecs, objp, imgp = get_intrinsics_from_checker(img_dir, (8, 6)) 
    np.save("robot_intrinsic.npy", mat) 
    print(mat) 
    print(rvecs) 
    print(tvecs) 
