import cv2
import numpy as np
import pyautogui


# init part
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') #save face casacade xml file to var
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml') #save eye casacade xml file to var
detector_params = cv2.SimpleBlobDetector_Params() #save cv2 blob detector to to var
detector_params.filterByArea = True #sets up opencv filter
detector_params.maxArea = 1500 #sets max area of eye recognition to 1500 pixels
detector = cv2.SimpleBlobDetector_create(detector_params) #defines detector by creating the blob detector within var


'''
Mac pixel dimensions: 1920 x 1080
Dimensions / 20?
'''


def detect_faces(img, cascade): 
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #converts img to gray, saves to var
    coords = cascade.detectMultiScale(gray_frame, 1.3, 5) #detectmultiscale used to detect facial features with cascade within the frame
    for (x,y,w,h) in coords: #cascade recognizes the x, y coordinates, width, and height of face in arrays
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,255),2) #create rectangle around face in response to coord values
    if len(coords) > 1: 
        biggest = (0, 0, 0, 0) #since many faces can be recognized within one frame, biggest limits it to the largest face
        for i in coords:
            if i[3] > biggest[3]:
                biggest = i
        biggest = np.array([i], np.int32)
    elif len(coords) == 1: #checks if one face is detected
        biggest = coords
    else:
        return None
    for (x, y, w, h) in biggest: #cuts out rest of image except frame
        frame = img[y:y + h, x:x + w]
    return frame


def detect_eyes(img, cascade):
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #makes frame gray
    #gray_face = gray_frame[y:y+h, x:x+w]
    eyes = cascade.detectMultiScale(gray_frame, 1.3, 5) #detects eyes
    for (ex,ey,ew,eh) in eyes: #identifies x, y coordinates, width and height of eyes and makes rectangles around them
        cv2.rectangle(img,(ex,ey),(ex+ew,ey+eh),(120,255,87),2)
    width = np.size(img, 1)  #get face frame width
    height = np.size(img, 0)  #get face frame height
    left_eye = None #predefines left_eye and right_eye so system does not crash if eyes are not found
    right_eye = None
    for (x, y, w, h) in eyes:
        if y > height / 2: #pass if eyes are at the bottom
            pass
        eyecenter = x + w / 2  #get the center of eye
        if eyecenter < width * 0.5: #seperates the two eyes and saves to seperate variables (only applies to blob detection)
            left_eye = img[y:y + h, x:x + w]
        else:
            right_eye = img[y:y + h, x:x + w]
    return left_eye, right_eye


def cut_eyebrows(img): #cuts out eyebrows so cs2 does not identify them as a pupil in blob detection
    height, width = img.shape[:2]
    eyebrow_h = int(height / 4)
    img = img[eyebrow_h:height, 0:width]  #cut eyebrows out

    return img



'''
Index:
[<KeyPoint 0x10145c930>]
[<KeyPoint 0x1015083c0>]
[<KeyPoint 0x10145c930>]
'''



def blob_process(img, threshold, detector): #threshold will be used to change the contrast of img so it can adjust to light in any setting
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #make img gray
    _, img = cv2.threshold(gray_frame, threshold, 255, cv2.THRESH_BINARY) #_, in this case, stands for an uneeded variable
    img = cv2.erode(img, None, iterations=2) #cv2.erode finds minimum amount of pixels, iterations: number of times erode is applied
    img = cv2.dilate(img, None, iterations=4) #dilate finds max object area. None is used because erode and dilate expect a second argument, but is not necessary
    img = cv2.medianBlur(img, 5) #medianBlur processes img by defining its outline and reducing noise
    keypoints = detector.detect(img)

    #x = keypoints[i].pt[0] #i is the index of the blob you want to get the position
    #y = keypoints[i].pt[1]
    #pyautogui.moveTo(200,150)
    

    print(keypoints)
    return keypoints


def nothing(x): #when using a trackbar in cv2, it requires a seperate function that executes the movement
    pass #however, since all it needs is a value, this simple function does the job


def main():
    cap = cv2.VideoCapture(0) #opens camera
    cv2.namedWindow('image')
    cv2.createTrackbar('threshold', 'image', 0, 255, nothing) #create trackbar from values 0 to 255
    while True: #while the code runs... 
    #inside while: essentially connects functions while camera runs and so it can be applied to theshold
        _, frame = cap.read() #bool statement that checks that frame is a valid input to run
        face_frame = detect_faces(frame, face_cascade)
        if face_frame is not None: #if not none is also here in case the user face is not detected (otherwise it'd crash)
            eyes = detect_eyes(face_frame, eye_cascade)
            for eye in eyes:
                if eye is not None: #if not none is also here in case the user blinks (otherwise it'd crash)
                    threshold = r = cv2.getTrackbarPos('threshold', 'image') 
                    eye = cut_eyebrows(eye)
                    keypoints = blob_process(eye, threshold, detector)
                    eye = cv2.drawKeypoints(eye, keypoints, eye, (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imshow('image', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): #0xFF == order('q') when q is pressed, wait 1 ms to break while loop
            break
    cap.release() #"releases" or (cv2's version of) returning video
    cv2.destroyAllWindows() #destroy windows that were previously open


if __name__ == "__main__": #__name__ is found in files run as .py scripts, meaning the function will be called when the file is run in terminal
    main()