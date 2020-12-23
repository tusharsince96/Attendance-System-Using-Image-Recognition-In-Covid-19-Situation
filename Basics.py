import cv2
import numpy as np
import face_recognition

imgElon = face_recognition.load_image_file('../ImageBasics/ElonMusk.jpg')
imgElon = cv2.cvtColor(imgElon,cv2.COLOR_BGR2RGB)
imgTest = face_recognition.load_image_file('../ImageBasics/BillGates.jpg')
imgTest = cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)

facloc = face_recognition.face_locations(imgElon)[0]
encodeElon = face_recognition.face_encodings(imgElon)[0]
cv2.rectangle(imgElon,(facloc[3],facloc[0]),(facloc[1],facloc[2]),(255,0,255),2)


faclocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest,(faclocTest[3],faclocTest[0]),(faclocTest[1],faclocTest[2]),(255,0,255),2)

results = face_recognition.compare_faces([encodeElon],encodeTest)
faceDis = face_recognition.face_distance([encodeElon],encodeTest)
print(results,faceDis)
cv2.putText(imgTest,f'{results} {round(faceDis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)


cv2.imshow('Elon Musk',imgElon)
cv2.imshow('Elon Test' ,imgTest)
cv2.waitKey(0)