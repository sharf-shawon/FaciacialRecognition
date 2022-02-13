import cv2
import face_recognition

image = cv2.imread("images/Sharfuddin Shawon.jpeg")         #Load an image
rgb_data = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)           #convert it to rgb
encoded_data = face_recognition.face_encodings(rgb_data)[0] #encode face data

image2 = cv2.imread("images/Sharfuddin Shawon.jpeg")         #Load an image
rgb_data2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)          #convert it to rgb
encoded_data2 = face_recognition.face_encodings(rgb_data2)[0]#encode face data

result = face_recognition.compare_faces([encoded_data], encoded_data2)
print("Both are same person" if result else "Different Persons")

cv2.imshow("First Image", image)
cv2.imshow("SeScond Image", image2)
cv2.waitKey(0)