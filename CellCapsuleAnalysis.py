import cv2
import numpy as np

def count_pixels_between_circles(image_path):
    # Read the image
    img = cv2.imread(image_path)

    # Convert image to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define range for red and green colors in HSV
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])

    lower_green = np.array([35, 100, 100])
    upper_green = np.array([85, 255, 255])

    # Create masks
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    # Find contours
    contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_green, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Assuming the largest contour for each color is the required circle, sort and pick the largest
    contour_red = max(contours_red, key=cv2.contourArea)
    contour_green = max(contours_green, key=cv2.contourArea)

    # Create a blank mask and draw the contours (filled) on them
    mask_red_filled = np.zeros_like(mask_red)
    cv2.drawContours(mask_red_filled, [contour_red], -1, (255), thickness=cv2.FILLED)

    mask_green_filled = np.zeros_like(mask_green)
    cv2.drawContours(mask_green_filled, [contour_green], -1, (255), thickness=cv2.FILLED)

    # Subtract the green mask from the red mask to get the region between the circles
    between_circles = mask_red_filled - mask_green_filled

    # show the image with the region between the circles highlighted in red with a black background
    cv2.imshow('between_circles', between_circles)


    # Count the non-zero pixels
    count = cv2.countNonZero(between_circles)

    return count

def is_rectangle(cnt, epsilon_factor=0.02):
    # Approximate the contour to determine its shape
    epsilon = epsilon_factor * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)

    # If the shape has 4 vertices, it's potentially a rectangle
    return len(approx) == 4

def measure_black_rectangle_width(image_path):
    img = cv2.imread(image_path)

    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Threshold to isolate black objects
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)  

    # Find black contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Iterate over each contour to find the rectangle
    for cnt in contours:
        if is_rectangle(cnt):
            # Get the width of the bounding rectangle of the contour
            _, _, width, _ = cv2.boundingRect(cnt)
            return width

    # If no rectangle is found
    return 0

# Test the function
image_path = '/Users/Barney/Documents/Screenshots/AnnotatedSample.png'

areaInPixels = count_pixels_between_circles(image_path)
pixelWidthOfBlackRectangle = measure_black_rectangle_width(image_path)

blackBoxWidthInMicrons = 0.1
pixelsInOneMicron = pixelWidthOfBlackRectangle / blackBoxWidthInMicrons

# Calculate area in um^2
areaInUm = areaInPixels / pixelsInOneMicron**2

# Print results
print(f"Area in Pixels: {areaInPixels}")
print(f"Pixels in one micron: {pixelsInOneMicron}")
print(f"Area in um^2: {areaInUm}")
