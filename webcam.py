import cv2


xml_haar_cascade = 'haarcascade_frontalface_default.xml'

# carrefar classificador
face_classifier = cv2.CascadeClassifier(xml_haar_cascade)

# Iniciar Camera
catch = cv2.VideoCapture(0)
catch.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
catch.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

while not cv2.waitKey(20) & 0xff == ord('q'):
    rect, frame_color = catch.read()
    gray = cv2.cvtColor(frame_color, cv2.COLOR_BGR2GRAY)

    faces = face_classifier.detectMultiScale(gray)
    for x, y, w, h in faces:
        cv2.rectangle(frame_color, (x, y), (x + w, y + h), (0,0,255), 2)

    cv2.imshow('Color', frame_color)


catch.release()
cv2.destroyAllWindows()