#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Object Detection using torch, pretrained SSD neural network (person & dog classes), 
OpenCV to detact people and dogs through video.

Filename: object_detection.py
"""

# Object Detection

# Importing the libraries
import torch
from torch.autograd import Variable
import cv2
from data import BaseTransform, VOC_CLASSES as labelmap
from ssd import build_ssd
import imageio
import timeit # just to check elaspsed time 

#  DEFINE A DETECT FUNCTION that will take as inputs, a frame,
#  a ssd neural network, and a transformation to be applied on the images, 
#  and that will return the frame with the detector rectangle.
def detect(frame, net, transform): 
    # get the height and the width of the frame.
    height, width = frame.shape[:2] 
    # apply the transformation to our frame.
    frame_t = transform(frame)[0] 
    # convert the frame into a torch tensor.
    x = torch.from_numpy(frame_t).permute(2, 0, 1) 
    # add a fake dimension corresponding to the batch.
    x = Variable(x.unsqueeze(0)) 
    # feed the neural network ssd with the image and we get the output y.
    y = net(x) 
    # create the detections tensor contained in the output y.
    detections = y.data 
    # create a tensor object of dimensions [width, height, width, height].
    scale = torch.Tensor([width, height, width, height]) 
    
    # For every class:
    for i in range(detections.size(1)): 
        # initialize the loop variable j that will correspond 
        # to the occurrences of the class.
        j = 0 
        # take into account all the occurrences j of the class i 
        # that have a matching score larger than 0.6.
        while detections[0, i, j, 0] >= 0.6:
            # get the coordinates of the points at the upper left and 
            # the lower right of the detector rectangle.
            pt = (detections[0, i, j, 1:] * scale).numpy() 
            # draw a rectangle around the detected object.
            cv2.rectangle(frame, (int(pt[0]), int(pt[1])), (int(pt[2]), int(pt[3])), (255, 0, 0), 2) 
            # put the label of the class right above the rectangle.
            cv2.putText(frame, labelmap[i - 1], (int(pt[0]), int(pt[1])), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA) 
            # increment j to get to the next occurrence.
            j += 1 
    # return the original frame with the detector rectangle and 
    # the label around the detected object.
    return frame 

# CREATING THE SSD NEURAL NETWORK
# create an object that is our neural network ssd.
net = build_ssd('test') 
# get the weights of the neural network from another one 
# that is pretrained (ssd300_mAP_77.43_v2.pth).
net.load_state_dict(torch.load('ssd300_mAP_77.43_v2.pth', map_location = lambda storage, loc: storage)) 

# CREATING THE TRANSFORMATION
# create an object of the BaseTransform class, a class that will do the 
# required transformations so that the image can be the input of the neural network.
transform = BaseTransform(net.size, (104/256.0, 117/256.0, 123/256.0)) 

# DOING OBJECT DETECTION ON A VIDEO
# variables for video and output files
vFilename = 'funny_dog.mp4'
vFilename_output = 'funny_dog_output.mp4'

# Time Starts
start_time = timeit.default_timer()

# open the video.
print('Opening the video: ', vFilename)
reader = imageio.get_reader(vFilename) 

# get the fps frequence (frames per second).
fps = reader.get_meta_data()['fps'] 
# create an output video with this same fps frequence.
writer = imageio.get_writer(vFilename_output, fps = fps)
print('Created the output video: ', vFilename_output)

# iterate on the frames of the output video: 
for i, frame in enumerate(reader): 
    # call our detect function to detect the object on the frame.
    frame = detect(frame, net.eval(), transform) 
    # add the next frame in the output video.
    writer.append_data(frame) 
    # print the number of the processed frame.
    print(i) 
    
# close the process that handles the creation of the output video.
writer.close() 

# Time ends
elapsed = timeit.default_timer() - start_time
print('Detection completed! Detection time: ', elapsed)
