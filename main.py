import cv2
import matplotlib.pyplot as plt



# path='group.jpg'

# image=cv2.imread(path)

# gray_image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
# dst=cv2.equalizeHist(gray_image)


# face_classifier = cv2.CascadeClassifier(
#     cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
# )

# face = face_classifier.detectMultiScale(
#     gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
# )

# for (x, y, w, h) in face:
#     cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 4)

# img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# cv2.imshow("face detect",img_rgb)

# cv2.waitKey(0)

cap = cv2.VideoCapture(0)

face_classifier=cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")

def detect_bounding_box(vid):
    gray_image = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))
    for (x, y, w, h) in faces:
        cv2.rectangle(vid, (x, y), (x + w, y + h), (0, 255, 0), 4)
    return vid

while True:

    ret,frame=cap.read()

    frame=detect_bounding_box(frame)
    # img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    cv2.imshow("face detect",frame)


    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()