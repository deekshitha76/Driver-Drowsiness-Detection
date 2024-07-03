from scipy.spatial import distance
from imutils import face_utils
from pygame import mixer
import imutils
import dlib
import cv2

# Initialize mixer for playing sound
mixer.init()
mixer.music.load("C:/Users/Hp/Desktop/iot/music.wav")

def eye_aspect_ratio(eye):
    # Compute the euclidean distances between the vertical eye landmarks
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    # Compute the euclidean distance between the horizontal eye landmarks
    C = distance.euclidean(eye[0], eye[3])
    # Compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)
    return ear

# Threshold for eye aspect ratio to determine drowsiness
thresh = 0.25
# Consecutive frame count for which the threshold must be below to raise an alert
frame_check = 20

# Initialize dlib's face detector (HOG-based) and then create the facial landmark predictor
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("C:/Users/Hp/Desktop/iot/shape_predictor_68_face_landmarks.dat")

# Grab the indexes of the facial landmarks for the left and right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

# Start the video capture
cap = cv2.VideoCapture(0)
flag = 0

if not cap.isOpened():
    print("Error: Could not open video capture.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Resize the frame to have a maximum width of 450 pixels
    frame = imutils.resize(frame, width=450)
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Ensure the gray image is in the correct format
    if gray.dtype != 'uint8':
        gray = gray.astype('uint8')

    # Detect faces in the grayscale frame
    subjects = detect(gray, 0)
    for subject in subjects:
        # Determine the facial landmarks for the face region, then convert the facial landmark (x, y)-coordinates to a NumPy array
        shape = predict(gray, subject)
        shape = face_utils.shape_to_np(shape)

        # Extract the left and right eye coordinates
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]

        # Compute the eye aspect ratio for both eyes
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        # Average the eye aspect ratio together for both eyes
        ear = (leftEAR + rightEAR) / 2.0

        # Compute the convex hull for the left and right eye, then visualize each of the eyes
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        # Check to see if the eye aspect ratio is below the blink threshold, and if so, increment the blink frame counter
        if ear < thresh:
            flag += 1
            if flag >= frame_check:
                cv2.putText(frame, "*ALERT!*", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "*ALERT!*", (10, 325),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                if not mixer.music.get_busy():  # Ensure the alert sound is played only once
                    mixer.music.play()
        else:
            flag = 0
            mixer.music.stop()  # Stop the alert sound when not needed

    # Display the frame
    cv2.imshow("Frame", frame)

    # If the q key was pressed, break from the loop
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# Do a bit of cleanup
cv2.destroyAllWindows()
cap.release()