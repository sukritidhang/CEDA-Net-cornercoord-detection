#Import required libraries

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import glob
import skimage.io
from skimage import io
import natsort
import csv
import imutils

#class creation

class detect_four_corner(object):
    #class contructor
    def __init__(self, gt_image):#, image_name):
        self.gt_image = gt_image
        #self.image_name = image_name

    def grayimg(self):
        self.gray_array = []
        self.filename_array = []      
        for i in range(len(self.gt_image)):

            self.image_name = read_files[i]
            #print(image_name)
            self.textx_file_name = os.path.basename(self.image_name)
            print('test x:', self.textx_file_name)
            
            #print(self.pred_image[i].shape)
            #convertGrayscale
            self.gray_images = cv2.cvtColor(self.gt_image[i], cv2.COLOR_BGR2GRAY)
            self.gray_images = cv2.GaussianBlur(self.gray_images, (5,5), 0)
            print(self.gray_images.shape)
            
            self.gray_array.append(self.gray_images)
            self.filename_array.append(self.textx_file_name)
            

    
    def sobeledgeddetect(self):

        print(f"lenof grayimages:{len(self.gray_array)}")
        for i in range(len(self.gt_image)):
            
            self.sobel_x = cv2.Sobel(self.gray_array[i], cv2.CV_64F, 1, 0, ksize=3)
            self.sobel_y = cv2.Sobel(self.gray_array[i], cv2.CV_64F, 0, 1, ksize=3)
            self.sobel_magnitude = np.sqrt(self.sobel_x**2 + self.sobel_y**2)
            

            print('shape of sobel_magnitude:', self.sobel_magnitude.shape)
            print('image data type:', self.sobel_magnitude.dtype)
            print('image value range:', self.sobel_magnitude.min(), self.sobel_magnitude.max())

            #self.sobel_magnitude = np.uint8((self.sobel_magnitude / self.sobel_magnitude.max()) * 255.0)
            
            # Convert gradient magnitude to a suitable binary format
            _, self.binary_image = cv2.threshold(self.sobel_magnitude, 0, 255, cv2.THRESH_BINARY)

            # Find contours on the binary image
            contours, _ = cv2.findContours(self.binary_image.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            
            #contours = cv2.findContours(self.sobel_magnitude, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            #cnts = imutils.grab_contours(contours)
            #c = max(cnts, key=cv2.contourArea)

            x, y, w, h = cv2.boundingRect(self.gray_array[i])
            imagefilename = self.filename_array[i]
            
            print('image name:', imagefilename)            
            print('x:',x, '|', 'y:',y )
            print('w:',w, '|', 'h:',h)
            self.gt_imagec = cv2.rectangle(self.gt_image[i], (x,y), (x+w, y+h), (0,0,0),2)

            top_left = (x, y)
            top_right = (x+w,y)
            bottom_left = (x, y+h)
            bottom_right = (x+w, y+h)
            #x, y coordinate
            left_top_x1 = x
            left_top_y1 = y
            right_top_x2 = x+w
            right_top_y2 = y
            left_bottom_x3 = x
            left_bottom_y3 = y+h
            right_bottom_x4 = x+w
            right_bottom_y4 = y+h

            #self.gt_imagec = cv2.drawContours(self.gt_image[i], [c], -1, (0, 255, 255), 2)
            
            self.gt_imagec = cv2.circle(self.gt_image[i], top_left, 2, (0, 0, 255), 2)
            self.gt_imagec = cv2.circle(self.gt_image[i], bottom_right, 2, (0, 255, 0), 2)
            self.gt_imagec = cv2.circle(self.gt_image[i], top_right, 2, (255, 0, 0), 2)
            self.gt_imagec = cv2.circle(self.gt_image[i], bottom_left, 2, (0, 255, 255), 2)

            #self.gt_imagec = cv2.cvtColor(self.gt_imagec, cv2.COLOR_GRAY2BGR)
            
            print(self.gt_imagec.shape)
            
            cv2.imshow('gt_four_corner', self.gt_imagec)
            #Method to write the detected four corner
            cv2.imwrite(f'E:/python_programming/phd/billboard_edgedetect/results/sobel/gt_4corner_sobel/{imagefilename}', self.gt_imagec)

            print("left-top:",top_left, '|', "right-top:",top_right)#red | blue
            print("right-bottom:",bottom_right, '|', "left-bottom:",bottom_left)#green | teal

            
            # open the file in the write mode
            with open('E:/python_programming/phd/billboard_edgedetect/results/sobel/gt_4corner_sobel.csv','a', newline = '') as f:
                writer = csv.writer(f)#create the csv writer
                data = [imagefilename, top_left, top_right, bottom_right, bottom_left, left_top_x1, left_top_y1, right_top_x2, right_top_y2, left_bottom_x3, left_bottom_y3, right_bottom_x4, right_bottom_y4]
                writer.writerow(data)# write a row to the csv file
        

    

# Main method

if __name__ == "__main__":

    gt_image = "E:/python_programming/phd/billboard_edgedetect/dataset/mask/"


    num_images = len(os.listdir(gt_image))
    print('num of images:', num_images)

    read_files =  natsort.natsorted(glob.glob(gt_image + "*.jpg"))

    #print(read_files)
          
    gt = [cv2.imread(file)
                for file in read_files]


          
    header = ['image_name', 'left-top', 'right-top', 'right-bottom', 'left-bottom', 'left_top_x1', 'left_top_y1', 'right_top_x2', 'right_top_y2', 'left_bottom_x3', 'left_bottom_y3', 'right_bottom_x4', 'right_bottom_y4']
                                   
    # open the file in the write mode
    with open('E:/python_programming/phd/billboard_edgedetect/results/sobel/gt_4corner_sobel.csv','w', newline = '') as f:
            # create the csv writer
            writer = csv.writer(f)
            # write a row to the csv file
            writer.writerow(header)

  
    
    detect4corner = detect_four_corner(gt)

    detect4corner.grayimg()
    detect4corner.sobeledgeddetect()

