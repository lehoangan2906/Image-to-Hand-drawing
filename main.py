import cv2 as cv
import numpy as np
from edges import Edges  
from PIL import Image, ImageOps

filename = "photo.jpg"

class Sketch:
    def __init__ (self, image):
        self.img = cv.imread(image)

    def sketch(self):
        # Convert the image to grayscale
        grey = cv.cvtColor(self.img, cv.COLOR_BGR2GRAY)
        
        # Invert the grayscale image
        inv = 255 - grey
        
        # Apply Gaussian blur to the inverted image
        blur = cv.GaussianBlur(inv, (13, 13), 0)
        
        # Obtain the sketch effect by Dodge Blend technique: dividing the grayscale image by the inverse blur
        return cv.divide(grey, 255-blur, scale=256)


img = Image.open(filename)

# Create a sketch and save it
sketch = Sketch(filename).sketch()
cv.imwrite("sketch.png", sketch)

# Perform edge detection using Canny edge detection algorithm
edges = Edges(filename).edges()

# Apply the edges to the sketch
# cv.biwise_and(image1, image2, mask image that defines which pixels to include in the operation)
sketch = cv.bitwise_and(sketch, edges, edges)

# Apply thresholding to the sketch
(thresh, sketch) = cv.threshold(sketch, 240, 255, cv.THRESH_BINARY)

# converts gray scale image - sketch to a 4-channel RGBA (Red, Green, Blue, Alpha) color image
sketchColor = cv.cvtColor(sketch, cv.COLOR_GRAY2RGBA)

# Save the final image with transparency
cv.imwrite("final.png", sketchColor)

# Create a PIL Image from the final image and display it
final = Image.fromarray(sketchColor)
final.show()
