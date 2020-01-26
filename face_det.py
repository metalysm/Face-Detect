import cv2
import imageio


face_cascade = cv2.CascadeClassifier('haarcascade-frontalface-default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade-eye.xml')

def detect(frame):
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5) #faces tapıllardan oluşuyor,bunlarda koor. olan x ve y var.
    for(x,y,w,h) in faces:        #w genişlik,h yükselik , x y koordinat
        cv2.rectangle(frame, (x,y),(x+w,y+h),(255,0,0),2)  #RGB kodları 255 0 0 yani red green blue. 0 hiç olmadığı 255 ise tam olduğu anlamına gelir.
        # 2 ise ikdötgenlerin kalınlığı. x+w y+h sağ alta köşe koordinat.
        gray_face = gray[y:y+h,x:x+w] #y den y+h a kadar olan bölgeyi alma
        color_face = frame[y:y+h,x:x+w]
        eyes = eye_cascade.detectMultiScale(gray_face,1.1,3)
        for(ex,ey,ew,eh) in eyes:
            cv2.rectangle(color_face,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    return frame

reader = imageio.get_reader('1.mp4')
fps = reader.get_meta_data()['fps']
writer = imageio.get_writer('output.mp4',fps=fps)                                 #fps aynı fps değeriyle kalıyor.
for i ,frame in enumerate(reader):      #reader için aldığımız her frame de for döngüsü dönecek.
    frame = detect(frame)
    writer.append_data(frame)
    print(i)
writer.close()




