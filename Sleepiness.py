import dlib
import cv2
import numpy as np
from scipy.spatial import distance as dist

cap = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


def drawLine(start, end):
    cv2.line(frame, start, end, (0, 0, 255), 2)


while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    for face in faces:
        x1, y1 = face.left(), face.top()
        x2, y2 = face.right(), face.bottom()

        landmarks = predictor(gray, face)

        myPoints = []
        for n in range(68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y

            myPoints.append([x, y])

        myPoints = np.array(myPoints)

        p2 = tuple(myPoints[50])
        p6 = tuple(myPoints[58])

        p3 = tuple(myPoints[52])
        p5 = tuple(myPoints[56])

        p1 = tuple(myPoints[48])
        p4 = tuple(myPoints[54])

        A = dist.euclidean(p2, p6)
        B = dist.euclidean(p3, p5)
        C = dist.euclidean(p1, p4)
        aspect_ratio = (A + B) / (2.0 * C)

        drawLine(p1, p2)
        drawLine(p2, p3)
        drawLine(p3, p4)
        drawLine(p4, p5)
        drawLine(p5, p6)
        drawLine(p6, p1)

        if aspect_ratio >= 0.7:
            cv2.putText(
                frame,
                "Sleepiness detected",
                (50, 50),
                cv2.FONT_HERSHEY_COMPLEX,
                1,
                (0, 0, 255),
                2,
            )

    cv2.imshow("frame", frame)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()


"""
p2-p6 + p3-p5 / 2(p1-p4)
"""
