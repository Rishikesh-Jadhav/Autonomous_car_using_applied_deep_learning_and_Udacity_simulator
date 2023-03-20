import cv2
import numpy as np
import matplotlib.pyplot as plt

# In a nutshell -
# 1- grayscale conversion
# 2- Blurrring to remove noise
# 3- Removing noise
# 4- Creating a mask for the region of interest
# 5- Bitwise adding created mask and canny image to highlight the region of region_of_interest
# 6- Writing a funtion to create lines at edges detected
# 7- first check the lines are present
# 7-  at by joining the white dots on a black backgound 


def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    canny = cv2.Canny(blur,50,150)
    return canny



def region_of_interest(image):
    height  = image.shape[0]
    polygons = np.array([[(200,height), (1100,height), (550,250)]])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask,polygons,255)
    masked_image = cv2.bitwise_and(image, mask)

    return masked_image

# using ths function to detect lines and dray them on a newly created black bacground
def display_lines(image,lines):
    line_image=np.zeros_like(image)
    #to check if the lines have been detected or not we have to check if the array is not empty
    if lines is not None:
#condition for not empty
        for line in lines:
            #print(line) 
#after checking lines note that each line is a 2d array with 1 row and 4 cols
# now we have to reshape everything to a 1 D 
            x1,y1,x2,y2 = line.reshape(4) #check syntax
            cv2.line(line_image, (x1,y1), (x2,y2), (255,0,0), 5)#1st arg- draw line on what? (the black image we created), 2nd & 3rd arg - in which co-oridinates we want to draw the lines 4thh arg is the color of thhe line, 5th is the line thickness
    return line_image



#Optimized code snippet
#for x1,y1,x2,y2 in lines : 
#   cv2.line(line_image, (x1,y1), (x2,y2), (255,0,0), 5)#1st arg- draw line on what? (the black image we created), 2nd & 3rd arg - in which co-oridinates we want to draw the lines 4thh arg is the color of thhe line, 5th is the line thickness




###OPTIMIZING(last step)
def average_slope_intercept(image,lines):
    left_fit = [] #co-ordinates of averaged lines on the left
    right_fit = [] #co-ordinates of averaged lines on the left
    for line in lines:
        x1,y1,x2,y2 = line.reshape(4) #unpaacking lines in 4 co-ordinates

        parameters = np.polyfit((x1,x2), (y1,y2), 1) #polyfit?- gives slope and y intercept of entered x and y values from the reshaped array
        # after printing the parameters we get 2 values where slope is at index zero and yintercept at index 1
        slope = parameters[0]
        intercept = parameters[1]
        # now the given slope of the line is on left side or right side of the image?
        # lines on left have negative slope and on left have positive slope
        if slope<0:
            left_fit.append((slope,intercept))
        else:
            right_fit.append((slope,intercept))
    print(left_fit)
    print(right_fit)   
    #now average all values in single slope and y intercept

    left_fit_average = np.average(left_fit,axis=0)    #average traverse vertically(axis = 0) for slope and yint average
    right_fit_average = np.average(right_fit,axis=0)

    print(left_fit_average, 'left') #average slope of left side lines
    print(right_fit_average, 'right') #average slope of right side lines

    left_line = make_coordinates(image,left_fit_average)
    right_line = make_coordinates(image,right_fit_average)

    return np.array([left_line,right_line])

##now that we have our slopes and intercepts of average lines to be placed 
# we need co-ordinates to place them at

def make_coordinates(image,line_parameters):
    slope,intercept = line_parameters
    #now hat we have slope and inteercept we need the x1y1x2y2
    print(image.shape)
    ##printed the shape to understand the height = 704
    y1 = image.shape[0] # height
    y2 = int(y1*(3/5)) #3/5 because ny the looks of the image it seemed so
    x1 = int((y1 - intercept)/slope) #y = mx+b
    x2 = int((y2 - intercept)/slope) #y = mx+b

    return np.array([x1,y1,x2,y2])



image = cv2.imread('test_image.jpg')
lane_image = np.copy(image)
canny_image = canny(lane_image)

cropped_image = region_of_interest(canny_image)

# P = row = 2 pixels length ,theta-1 degree in radians Last parameter is the thresholds(minimum no of votes needed to accept a candidate line), 5th argument is a placeholder array(empty array), 6th arg is the minimum length of the array wel accept as a line, 7th is max line gap between segmented lines
lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap= 5)  

averaged_lines = average_slope_intercept(lane_image,lines)



#instead of the lines we created if we pass averaged lines
line_image = display_lines(lane_image, averaged_lines)

#next step is to Blend this image into the original image

###Make sure that the images you add are the same shape!!!
combined_image = cv2.addWeighted(lane_image,0.8, line_image,1, 1)#(img1, weight(multplied with each pixel of the image),img2,weight of img2, gamma value )
# sum of coloured image with line image
# as we created a black c=background with a pixel intensty with zeros adding it to any pixel intensity would not change the intensity of the combined image 
# Only when we blend the pixel intensities of the lines_image witth the oiginal will we be able to see the difference  
cv2.imshow('results',combined_image)
cv2.waitKey(0)

