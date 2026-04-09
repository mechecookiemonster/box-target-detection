import cv2
import numpy as np


### CONFIGURATION STEP ###
# Cam
CAMERA_ID = 0 ######### ! to be changed for Pi cam

# Box detection by color
BOX_COLOR_LOWER = np.array([0, 180, 180]) ###### red for now
BOX_COLOR_UPPER = np.array([10, 255, 255])
BOX_MIN_AREA = 1000
BOX_ASPECT_MIN = 0.5
BOX_ASPECT_MAX = 2.0

# Aruco
ARUCO_DICT_TYPE = cv2.aruco.DICT_4X4_50

# Display
SHOW_MASK = True
SHOW_DISTANCE = True

### CONFIGURATION STEP - SEE ABOVE ###

# Initialise camera
cap = cv2.VideoCapture(0)

# Aruco dictionary:
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco_params = cv2.aruco.DetectorParameters()

# Color ranges
lower = np.array([0, 180, 180])
upper = np.array([10, 255, 255])

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    #Convert to hsv
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)


    # Box detection

    box_mask = cv2.inRange(hsv, lower, upper)
    box_contours, _ = cv2.findContours(box_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    box_position = None

    for contour in box_contours:
        area = cv2.contourArea(contour)
        print(f"Countour area: {area}")

        if area > 1000:
            x, y, w, h = cv2.boundingRect(contour)
            print(f"Drawing box at: ({x}, {y}, {w}, {h})")

            #Filter by shape
            aspect_ratio = w/float(h)
            if 0.5 < aspect_ratio < 2.0:
                #Draw green box around:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

                #Calculate center
                cx = x + w//2
                cy = y + h//2
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

                #Label
                cv2.putText(frame, "BOX", (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                box_position = (cx, cy)
                print(f"Box detected at: ({cx}, {cy})")

    #Target detection
    # Detect Aruco markers
    corners, ids, rejected = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=aruco_params)

    target_position = None

    # Draw detected markers
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        # Get center of first marker
        if len(corners) > 0:
            corner = corners[0][0]
            tx = int(corner[:, 0].mean())
            ty = int(corner[:, 1].mean())

            #draw blue dot at the center:
            cv2.circle(frame, (tx, ty), 5, (255, 0, 0), -1)
            print(f"Target at ({tx}, {ty})")

            #Label
            cv2.putText(frame, "TARGET", (tx-30, ty-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            target_position = (tx, ty)
            print(f"Target detected at ({tx}, {ty})")

    # Calculate distance (!both detected!) and draw line between
    if box_position and target_position:
        cv2.line(frame, box_position, target_position, (0, 255, 255), 2)

        # Calculate pixel distance
        dx = target_position[0] - box_position[0]
        dy = target_position[1] - box_position[1]
        distance = int(np.sqrt(dx**2 + dy**2))

        # Display distance
        mid_x = (box_position[0] + target_position[0]) // 2
        mid_y = (box_position[1] + target_position[1]) // 2
        cv2.putText(frame, f"{distance}px", (mid_x, mid_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        print(f"Distance: {distance} pixels")

    # Show results
    cv2.imshow('Box and Target Detection', frame)
    
    #cv2.imshow('Detection', frame)
    #cv2.imshow('Mask', mask)

    ## cv2.imshow('Camera Feed', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
