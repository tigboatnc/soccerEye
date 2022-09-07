from pathlib import Path
import cv2 
import numpy as np

def util_searching_all_files(directory: Path):   
    file_list = [] 
    for x in directory.iterdir():
        if x.is_file():
            if '.mp4' in str(x):
                file_list.append(x)
        else:
            file_list.append(util_searching_all_files(directory/x))
    return file_list

def stackImages(imgArray,axis=1):
    imStack = []
    for im in imgArray:

        if len(im.shape) == 2:
            imStack.append(cv2.cvtColor(im,cv2.COLOR_GRAY2RGB))
        else:
            imStack.append(im)
            
    return np.concatenate(imStack,axis=axis)