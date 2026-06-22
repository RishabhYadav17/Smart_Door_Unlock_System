import cv2
import os 

if not os.path.exists('dataset'):
    os.makedirs('dataset')

face_detector= cv2.CascadeClassifier(
    cv2.data.haarcascades+"haarcascade_frontalface_default.xml"
)

camera=cv2.VideoCapture(0)
count=0
while True:
    success,frame=camera.read()
    frame,
    gray=cv2.cvtColor(
    frame,
    cv2.COLOR_BGR2GRAY
    )
    faces=face_detector.detectMultiScale(
    gray,
    scaleFactor=1.5,
    minNeighbors=5
    )
    for (x, y, w, h) in faces:
      cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
      count=count+1
      face_img = gray[y:y+h, x:x+w]
      cv2.imwrite(f"dataset/User.1.{count}.jpg",face_img)
      print(f"Captured Image {count}/30")
      cv2.waitKey(100)
    cv2.imshow("CapturingFaces", frame)

    key = cv2.waitKey(1)
    if key == 27 or count >= 30:
        break

print("Successfully taken 30 photos for model training")

camera.release()
cv2.destroyAllWindows()
   