import cv2 
import numpy as np 
import torch
import numpy as np
import PIL
from PIL import Image
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
from matplotlib import pyplot as plt
from tqdm import tqdm
import models 


import tensorflow as tf
import pathlib
import numpy as np
from PIL import Image
import pandas as pd


# Field Isolation 

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
    

def P_IF_4_FPN(data,CHECKPOINT_LOCATION,inputType = 'single',device='cpu'):
    # data can be an image : Image.open('imgaddress')
    # or data can be a list of similarly formatted images 
    # device = cpu | cuda 

    if inputType == 'single':

        checkpoint = torch.load(CHECKPOINT_LOCATION,map_location=torch.device(device) )
        model = models.FPN_FieldMask_1("FPN", "resnet34", in_channels=3, out_classes=1)
        model.load_state_dict(checkpoint['state_dict'])

        image = models.FPN_FieldMask_1_preprocess(data,inputType)

        with torch.no_grad():
            model.eval()
            logits = model(image)
        
        pr_mask = logits.sigmoid()
        return pr_mask.numpy().squeeze()
    
    if inputType == 'list':

        checkpoint = torch.load(CHECKPOINT_LOCATION,map_location=torch.device(device) )
        model = models.FPN_FieldMask_1("FPN", "resnet34", in_channels=3, out_classes=1)
        model.load_state_dict(checkpoint['state_dict'])

        datalist = models.FPN_FieldMask_1_preprocess(data,inputType)
        OP = []

        with torch.no_grad():
            model.eval()

            for image in datalist:
                logits = model(image)
                pr_mask = logits.sigmoid()
                OP.append(pr_mask.numpy().squeeze())
        
        return OP




def P_IF_4_FPN_CV2(data,CHECKPOINT_LOCATION,inputType = 'single',device='cpu'):
    # data has to be an image from opencv typeset 
    # uniformity function
    # or data can be a list of similarly formatted images 
    # device = cpu | cuda 

    if inputType == 'single':

        checkpoint = torch.load(CHECKPOINT_LOCATION,map_location=torch.device(device) )
        model = models.FPN_FieldMask_1("FPN", "resnet34", in_channels=3, out_classes=1)
        model.load_state_dict(checkpoint['state_dict'])
        
        # Convert to PIL 
        im_pil = Image.fromarray(data)
        
        image = models.FPN_FieldMask_1_preprocess(im_pil,inputType)

        with torch.no_grad():
            model.eval()
            logits = model(image)
        
        pr_mask = logits.sigmoid()
        return pr_mask.numpy().squeeze()
    
    if inputType == 'list':

        checkpoint = torch.load(CHECKPOINT_LOCATION,map_location=torch.device(device) )
        model = models.FPN_FieldMask_1("FPN", "resnet34", in_channels=3, out_classes=1)
        model.load_state_dict(checkpoint['state_dict'])
        
        data2 = [Image.fromarray(x) for x in data]
        datalist = models.FPN_FieldMask_1_preprocess(data2,inputType)
        OP = []

        with torch.no_grad():
            model.eval()

            for image in tqdm(datalist):
                
                logits = model(image)
                pr_mask = logits.sigmoid()
                OP.append(pr_mask.numpy().squeeze())
        
        return OP

def P_PI_1_OD(frame,model,accuracy=0.7):
  # detects based on tensorflow models 
  # frame : np image 
  # model : tf model 
  # accuracy : float
  image_np = frame.copy()
  im_height,im_width,_ = image_np.shape
  input_tensor = tf.convert_to_tensor(image_np)
  input_tensor = input_tensor[tf.newaxis,...]
  model_fn = model.signatures['serving_default']
  output_dict = model_fn(input_tensor)
  num_detections = int(output_dict.pop('num_detections'))
  output_dict = {key:value[0, :num_detections].numpy() 
                  for key,value in output_dict.items()}
  output_dict['num_detections'] = num_detections
  # detection_classes should be ints.
  output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
    
  op = []
  for i in range(num_detections):
    op.append(
        {
            'box':output_dict['detection_boxes'][i],
            'score':output_dict['detection_scores'][i],
            'class':output_dict['detection_classes'][i]
        }
    )

  op = pd.DataFrame(op)
  op = op[op['class']==1]
  op = op[op['score']>accuracy]
  boxes_pixel = []

  for ind,rows in op.iterrows():
    ymin, xmin, ymax, xmax = rows['box']
    (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                  ymin * im_height, ymax * im_height)
    tl = (int(left),int(top))
    br = (int(right),int(bottom))
    boxes_pixel.append([tl,br])

  op['box_pixel'] =boxes_pixel
  return op 


def P_PI_1_OD_BOX(frame,detectionDF,color=(180,0,0),thickness=3):
  imcpy = frame.copy()
  # Create boxes for detections 
  for i,row in detectionDF.iterrows():
    tl,br = row['box_pixel']
    imcpy = cv2.rectangle(imcpy, tl, br, color, thickness)
  
  return imcpy