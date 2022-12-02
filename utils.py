from pathlib import Path
import cv2 
import numpy as np
from tqdm import tqdm

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


def resizeFrame(scale_percent,frame):
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
    
    
def applyToVideo(function,cap,skip=1,stack_original=False):
    # Check if camera opened successfully
    if (cap.isOpened() == False): 
        print("Unable to read video")
        return 
    total_frames = cap.get(7)
    if stack_original == False:
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        out = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
        for framePick in tqdm(range(0,int(total_frames),skip)):
            cap.set(1, framePick)
            ret, frame = cap.read()
            output = function(frame)
            if len(output.shape) == 2 :
                output = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)
                out.write(output)
            elif output.shape[2] == 3:
                out.write(output)
            else:
                print('Misformed Format for writing to video')
                break
    else:
        scale_percent = 50
        frame_width_single = int(cap.get(3))
        frame_height_single = int(cap.get(4))
        
        frame_width = int(frame_width_single *scale_percent  / 100) * 2 
        frame_height = int(frame_height_single * scale_percent / 100)
        
        out = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
        for framePick in tqdm(range(0,int(total_frames),skip)):
            cap.set(1, framePick)
            ret, frame = cap.read()
            output = function(frame)
            if len(output.shape) == 2 :
                output = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)
                output_r = resizeFrame(50,output)
                frame_r = resizeFrame(50,frame)
                output = np.concatenate([output_r,frame_r],axis=1)
                
                out.write(output)
            elif output.shape[2] == 3:
                output_r = resizeFrame(scale_percent,output)
                frame_r = resizeFrame(scale_percent,frame)
                output = np.concatenate([output_r,frame_r],axis=1)
                out.write(output)
            else:
                print('Misformed Format for writing to video')
                break
    out.release()


    
def get_all_files_in_dir(directory: Path):   
    # Returns a list of all files in a directory, requires Path as I/P
    file_list = [] # A list for storing files existing in directories

    for x in directory.iterdir():
        if x.is_file():

           file_list.append(x)
        else:

           file_list.append(searching_all_files(directory/x))

    return file_list