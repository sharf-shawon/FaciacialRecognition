import cv2
from FaceRecog import FaceRecog

# Encode faces from a folder
sfr = FaceRecog()
sfr.load_encoding_images("images/")

# Load Default Camera
cap = cv2.VideoCapture(0)


while True:
    ret, frame = cap.read()

    # Detect Faces
    face_locations, face_names = sfr.detect_known_faces(frame)
    for face_loc, name in zip(face_locations, face_names):
        y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]

        cv2.putText(frame, name,(x1, y2 + 25), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 4)

    cv2.imshow("Real Time Face Recognition (Press Esc to exit)", frame)

    key = cv2.waitKey(1)
    if key == 27: #press Esc key to exit
        break

cap.release()
cv2.destroyAllWindows()