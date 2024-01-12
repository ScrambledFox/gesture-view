import cv2 as cv
import numpy as np
import keyboard

# Defines camera capture
cap = cv.VideoCapture(1, cv.CAP_DSHOW)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 960)

leftOffset = 240
colorRadius = 63*2
colorSpacing = 25
yAlign = 570
selectedColor = 2

colors = [
    #B G R
    (0,0,255),  #red 0
    (0,255,255), #yellow 1
    (0,255,0), #green 2
    (255,255,0), #cyan 3
    (255,0,0), #blue 4
    (255,0,255), #magenta 5
]

# Main loop
while True:
    # Get frame
    ret, frame = cap.read()
    # End loop logic
    if not ret:
        break
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

    if keyboard.is_pressed('1'):
        selectedColor = 0
    if keyboard.is_pressed('2'):
        selectedColor = 1
    if keyboard.is_pressed('3'):
        selectedColor = 2
    if keyboard.is_pressed('4'):
        selectedColor = 3
    if keyboard.is_pressed('5'):
        selectedColor = 4
    if keyboard.is_pressed('6'):
        selectedColor = 5

    if keyboard.is_pressed('d'):
        leftOffset += 10
    if keyboard.is_pressed('a'):
        leftOffset -= 10

    rX = colorRadius+leftOffset
    mX = rX+colorRadius+colorSpacing
    bX = mX+colorRadius+colorSpacing
    cX = bX+colorRadius+colorSpacing
    gX = cX+colorRadius+colorSpacing
    yX = gX+colorRadius+colorSpacing

    positions = [rX, mX, bX, cX, gX, yX]

    selectedColor = positions.index(min(positions, key=lambda x:abs(x-640)))

    # Frame manipulation
    image = frame
    image = cv.flip(image, 1)
    image = cv.rotate(image, 0)

    # Using contours
    upperLimit = np.array([255, 255, 255])
    lowerLimit = np.array([100, 100, 100])
    mask = cv.inRange(image, lowerLimit, upperLimit)
    contours, hierarchy = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    largestContour = max(contours, key=cv.contourArea)
    imageWithContours = image.copy()
    imageWithContours = cv.drawContours(imageWithContours, [largestContour], 0, colors[selectedColor], thickness=cv.FILLED)
    alpha = 0.5
    image = cv.addWeighted(image, 1-alpha, imageWithContours, alpha, 0)

    for i in range(6):
        size = 63
        borderColor = (0,0,0)
        if selectedColor == i:
            size = size + 10
            borderColor = (255, 255, 255)
        image = cv.circle(image, (positions[i], yAlign), size+2, borderColor, -1)
        image = cv.circle(image, (positions[i], yAlign), size, colors[i], -1)

    cv.imshow('Shirt detection', image)
    
cap.release()
cv.destroyAllWindows()