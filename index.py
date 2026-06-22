import cv2

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
recognizer=cv2.face.LBPHFaceRecognizer_create()
try:
    recognizer.read("trainer.yml")
except:
    print("Warning,trainer.yml file not found,please run train_model.py")
camera =cv2.VideoCapture(0)
print("Camera Started")
print("ESC to exit")
while True:
    success,frame=camera.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=face_detector.detectMultiScale(
        gray,
        scaleFactor=1.4,
        minNeighbors=5
    )
    for(x,y,w,h) in faces:
        face_img=gray[y:y+h,x:x+w]
        id,confidence=recognizer.predict(face_img)
        if confidence < 70:
            color=(0,255,0)
            text="DOOR Unlocked"
        else:
            color(0,0,255)
            text="Access Denied"
        cv2.rectangle(
            frame,
            (x,y),
            (x+w,y+h),
            color,
            3
        )
        cv2.putText (
            frame,
            text,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,2
        )
    cv2.imshow("Smart Door Unlock SyStem",frame)
    key=cv2.waitKey(1)
    if key==27:
        break
camera.relaese()
cv2.destroyAllWindows()