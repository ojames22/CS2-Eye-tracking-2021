import cv2
import numpy as np


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') #save face casacade xml file to var
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml') #save eye casacade xml file to var


def detect_faces(img, cascade): 

    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #converts img to gray, saves to var
    coords = cascade.detectMultiScale(gray_frame, 1.3, 5) #detectmultiscale used to detect facial features with cascade within the frame

    for (x,y,w,h) in coords: #cascade recognizes the x, y coordinates, width, and height of face in arrays
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2) #create rectangle around face in response to coord values

    if len(coords) > 1: 
        biggest = (0, 0, 0, 0) #since many objects can be recognized as faces within one frame, biggest limits it to the face
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
        cv2.rectangle(img,(ex,ey),(ex+ew,ey+eh),(0,225,255),2)


def cut_eyebrows(img): #cuts out eyebrows so cs2 does not identify them as a pupil in blob detection

    height, width = img.shape[:2]
    eyebrow_h = int(height / 4)
    img = img[eyebrow_h:height, 0:width] #cut eyebrows out

    return img


def main():

    cap = cv2.VideoCapture(0) #opens camera
    cv2.namedWindow('image')

    while True: #while the code runs... 
    #inside while: essentially connects functions while camera runs and so it can be applied to theshold
        _, frame = cap.read() #bool statement that checks that frame is a valid input to run
        face_frame = detect_faces(frame, face_cascade)

        if face_frame is not None: #if not none is also here in case the user face is not detected (otherwise it'd crash)
            eyes = detect_eyes(face_frame, eye_cascade)

        cv2.imshow('image', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'): #0xFF == order('q') when q is pressed, wait 1 ms to break while loop
            break
            
    cap.release() #"releases" or (cv2's version of) returning video
    cv2.destroyAllWindows() #destroy windows that were previously open


if __name__ == "__main__": #__name__ is found in files run as .py scripts, meaning the function will be called when the file is run in terminal
    main()