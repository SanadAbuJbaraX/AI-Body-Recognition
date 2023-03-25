#######################################################################################
#                           Made By Sanad AbuJbara                                    #
#                                                                                     #
#                               AI Body Recognition                                   #
#    Usage                                                                            #
#  1. Recognizes Face and marks it                                                    #
#  2. Recognizes Hand and marks it                                                    #
#  3.Recognizes Body Posture and marks it                                             #
#                                                                                     #
#                               Error Handling                                        #
# 1. Make Sure that all libaries are installed properly                               #
# 2. Set the OPENCV_VIDEOIO_PRIORITY_MSMF variable to 0 in the enviroment variables   #
#                                                                                     #
# NOTE program may not work properly for more than one                                #
# TODO Fix this ^                                                                     #
# Made in python 3.10.10                                                              #
# Contact Me at discord : SANAD#5640                                                  #
#######################################################################################


# imports
import mediapipe as mpe
import cv2
import pyautogui as pt



video = cv2.VideoCapture(0) # opens the camera and starts capturing 
mpeDraw = mpe.solutions.drawing_utils # this is a util that allows us to mark what we want on the frame
mpeHandsClass = mpe.solutions.hands # This is Hand Detect Class contains needed info
HandDetect = mpeHandsClass.Hands(max_num_hands=2) # Hands Class process the frame mainly 
mpeFaceDetectClass = mpe.solutions.face_mesh
faceDetect = mpeFaceDetectClass.FaceMesh()
mpePoseClass = mpe.solutions.pose
bodyPosture = mpePoseClass.Pose()
# Main loop
while True:
    _,frame = video.read() # reading the img stored in video property and saving to a var called frame
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB) # changing the mode of the frame because mpe can only process rgb images 
    
    
    # Detect Hands
    
    handsResult = HandDetect.process(frame) # process the frame and detect if there any hands
   
    if handsResult.multi_hand_landmarks: # checks if there are any hands
        
        for hand in handsResult.multi_hand_landmarks: # loops through the hands
            
            mpeDraw.draw_landmarks(frame, hand, mpeHandsClass.HAND_CONNECTIONS) # draw the hands on the frame
    
    
    # Detect Faces
    
    faceDetectResult = faceDetect.process(frame) # process the frame and detect faces

    if faceDetectResult.multi_face_landmarks: # check if there any faces
        for face in faceDetectResult.multi_face_landmarks: # loop through the faces
            mpeDraw.draw_landmarks(frame, face,mpeFaceDetectClass.FACEMESH_CONTOURS) # draw the face
    

    # Detect Body Posture
    postureResult = bodyPosture.process(frame)

    if postureResult.pose_landmarks:
        mpeDraw.draw_landmarks(frame,postureResult.pose_landmarks,mpePoseClass.POSE_CONNECTIONS)

    # Display Frame
    frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR) # return the frame to it is orignal colour so it can be displayed
    frame = cv2.resize(frame,pt.size()) # resize the frame to full screen
    frame = cv2.flip(frame,2) # flip the frame because normally it is flipped the wrong way
    cv2.imshow('Body Detect',frame) # display the frame
    key = cv2.waitKey(1) 
    if key == ord('q'):
        break  # if the user presses 'q' the program closes

cv2.destroyAllWindows() # closes the program
