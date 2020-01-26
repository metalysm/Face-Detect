import cv2
import imageio

# Cascade yükleme
face_cascade = cv2.CascadeClassifier('haarcascade-frontalface-default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade-eye.xml')

# Tanıma yapacak fonksiyon
def detect(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
    return frame

# Proje klasörü içerisindeki image.jpg dosyasında yüz ve göz tesbiti yapılıyor.
# Daha sonra cikis.jpg dosyasına yazılıyor.
# Dosya isimlerini değiştirebilirsiniz.
image = imageio.imread('2.jpeg')
image = detect(frame=image)
imageio.imwrite('cikis.jpg', image)
