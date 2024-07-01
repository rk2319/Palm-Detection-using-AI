import cv2
import numpy as np

# Function to detect palm and fingers
def detect_palm_and_fingers(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Blur the image to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Threshold the image to create a binary image
    _, thresh = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours in the binary image
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Initialize variables to store palm coordinates and fingers
    palm_center = None
    fingers = []
    
    # If contours are found
    if contours:
        # Find the contour with maximum area (assumed to be the palm)
        max_contour = max(contours, key=cv2.contourArea)
        
        # Find the center of the contour
        moments = cv2.moments(max_contour)
        if moments["m00"] != 0:
            palm_center_x = int(moments["m10"] / moments["m00"])
            palm_center_y = int(moments["m01"] / moments["m00"])
            palm_center = (palm_center_x, palm_center_y)
        
        # Find convex hull of the contour
        hull = cv2.convexHull(max_contour, returnPoints=False)
        
        # Draw the edges of the palm with a green line
        palm_edges = [max_contour[i][0] for i in hull]
        cv2.polylines(image, [np.array(palm_edges)], True, (0, 255, 0), 2)
        
        # Find convexity defects to detect fingers
        defects = cv2.convexityDefects(max_contour, hull)
        
        # If defects are found
        if defects is not None:
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                start = tuple(max_contour[s][0])
                end = tuple(max_contour[e][0])
                far = tuple(max_contour[f][0])
                
                # Check if defect is not too close to the palm
                if d / 256 > 15:  # Threshold for distance
                    fingers.append(far)
    
    return palm_center, fingers

# Load image
image = cv2.imread('img1.jpg')

# Detect palm and fingers
palm_center, fingers = detect_palm_and_fingers(image)

# Draw palm center
if palm_center:
    cv2.circle(image, palm_center, 5, (0, 255, 0), -1)

# Draw fingers and lines from palm center to fingers
for finger in fingers:
    cv2.circle(image, finger, 5, (0, 0, 255), -1)
    cv2.line(image, palm_center, finger, (0, 255, 0), 2)

# Resize output window
cv2.namedWindow('Palm and Finger Detection', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Palm and Finger Detection', 400, 400)

# Display the result
cv2.imshow('Palm and Finger Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
