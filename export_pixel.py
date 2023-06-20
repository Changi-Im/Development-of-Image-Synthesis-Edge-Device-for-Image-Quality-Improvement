import torch
import args_fusion as args
import numpy as np
from binary_fractions import Binary
import cv2

if __name__ == "__main__":
    img1 = cv2.imread("./images/5_UN.jpg", 0)
    img2 = cv2.imread("./images/5_OV.jpg", 0)
    
    patch1 = img1.reshape(-1).astype("float64")
    patch2 = img2.reshape(-1).astype("float64")
    
    source1 = np.array([])
    source2 = np.array([])
    final = np.array([])
    
    for i in range(1600):
        i1 = Binary(patch1[i]).components()[1]
        i2 = Binary(patch2[i]).components()[1]
        
        f1 = Binary(patch1[i]).components()[2]
        f2 = Binary(patch2[i]).components()[2]
        
        if f1 == "":
            f1 = "000000000000"
            
        if f2 == "":
            f2 = "000000000000"
        a = "0000000"+i1
        b = "0000000"+i2
        source1 = np.append(source1, "storage1[{}] = 14'b{}_{};".format(i, "0"+i1, f1[:11]))
        source2 = np.append(source2, "storage1[{}] = 14'b{}_{};".format(1600+i, "0"+i2, f2[:11]))
        
    source1 = np.insert(source1,0,"// image 1")
    source2 = np.insert(source2,0,"// image 2")
    
    final = np.concatenate((source1, source2))
    
    
    np.savetxt("./pixel(14bit).txt", final, fmt='%s')
    