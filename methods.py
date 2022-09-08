import cv2 
import numpy as np 


def P_IF_1(frame,kernel_size=80,hsv_sens=140):
    # I = BGR 
    # O = Binary 
    
    blurred = cv2.blur(frame, (kernel_size, kernel_size))
    
    frame_hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    lower_green = np.array([60 - hsv_sens, 40, 40])
    upper_green = np.array([60 + hsv_sens, 255, 255])
    
    mask = cv2.inRange(frame_hsv, lower_green, upper_green)
    return mask



def colorQuantize(frame,div=64):
    quantized = frame // div * div + div // 2
    return quantized 

    
    
def P_IF_2(frame,kernel_size=80,hsv_sens=140,quant=100):
    # I = BGR 
    # O = Binary 
    
    blurred = cv2.blur(frame, (kernel_size, kernel_size))
    frame_hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    lower_green = np.array([60 - hsv_sens, 100, 100])
    upper_green = np.array([60 + hsv_sens, 255, 255])
    
    frame_hsv = colorQuantize(frame_hsv,div=quant)
    
    mask = cv2.inRange(frame_hsv, lower_green, upper_green)
    return mask



from skimage import exposure
from skimage import data, img_as_float,img_as_ubyte
def P_IF_3(frame,kernel_size=80,hsv_sens=20,pix_size=90):
    # I = BGR 
    # O = Binary 
    
    
    blurred = cv2.blur(frame, (kernel_size, kernel_size))
    
    
    # Desired "pixelated" size
    w, h = (pix_size, pix_size)
    height, width = blurred.shape[:2]
    # Resize input to "pixelated" size
    temp = cv2.resize(blurred, (w, h), interpolation=cv2.INTER_LINEAR)
    # Initialize output image
    frame_pixelated = cv2.resize(temp, (width, height), interpolation=cv2.INTER_NEAREST)
    # Equalization
    img_eq = img_as_ubyte(exposure.equalize_hist(frame_pixelated))
    # img_adapteq = exposure.equalize_adapthist(frame_pixelated, clip_limit=0.03)
    
    frame_hsv = cv2.cvtColor(img_eq, cv2.COLOR_BGR2HSV)
    


    # # Adaptive Equalization
   


    lower_green = np.array([60 - hsv_sens, 40, 40])
    upper_green = np.array([60 + hsv_sens, 255, 255])
    mask = cv2.inRange(frame_hsv, lower_green, upper_green)
    return mask
    