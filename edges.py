import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from PIL import Image, ImageOps

# perform edge detection on an input image using the Canny edge detection algorithm

class Edges:
    def __init__(self, image):
        # Initialize the Edges class with an input image
        self.img = cv.imread(image)
        
        # Convert the input image to grayscale
        self.grey = cv.cvtColor(self.img, cv.COLOR_BGR2GRAY)
        
        # Define a 2x2 square kernel for image processing
        self.kernel = np.ones((2, 2), np.uint8)

    def edges(self):
        # Apply Canny edge detection to the grayscale image
        edges = cv.Canny(self.grey, 200, 300)
        
        # Apply Gaussian blur to the detected edges
        blr = cv.GaussianBlur(edges, (3, 3), 0)
        
        # Dilate the edges using a 2x2 kernel
        # Help to enhance the detected edges by making them thicker
        dil = cv.dilate(edges, self.kernel, iterations=1)
        
        # Invert the dilated image to get a white-on-black representation
        image_gray = 255 - dil
        
        # Convert the grayscale image to an RGBA image
        image_rgba = cv.cvtColor(image_gray, cv.COLOR_GRAY2RGBA)
        
        # Identify white pixels and set their alpha channel to 0 for transparency to overlay the detected edges on another image later
        white = np.all(image_rgba == [255, 255, 255, 255], axis=-1)
        image_rgba[white, -1] = 0
        
        # Save the processed edge image
        cv.imwrite("edges.png", image_gray)
        
        # Return the processed edge image
        return image_gray