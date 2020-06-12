import json
import cv2
from PIL import Image  
import PIL 
import os
#def lambda_handler(event,context):  


carregaFace = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
carregaOlho = cv2.CascadeClassifier('haarcascade_eye.xml')
fotos_clientes_upload = r'/home/thiagorocha/Área de Trabalho/THg/teste.jpg'
fotos_clientes_recognition = r'/home/thiagorocha/Área de Trabalho/THg/python/'

imagem = cv2.imread(fotos_clientes_upload)
imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
faces = carregaFace.detectMultiScale(imagemCinza)

for(x, y, l, a) in faces:
    imagem = cv2.rectangle(imagem, (x, y), (x + l, y + a), (255, 0, 255), 2)
    localOlho = imagem[y:y + a, x:x + l]
    
    localOlhoCinza = cv2.cvtColor(localOlho, cv2.COLOR_BGR2GRAY)   
    detectado = carregaOlho.detectMultiScale(localOlhoCinza)

    for(ox, oy, ol, oa) in detectado:
        cv2.rectangle(localOlho, (ox, oy), (ox + ol, oy + oa), (0, 255, 0), 2)
        

cv2.imshow("Detecta Face e os Olhos ", imagem)
#cv2.waitKey(0) espera precionar a tecla 0 pra fechar janela
cv2.imwrite(os.path.join(fotos_clientes_recognition,"foto_cliente_recognition.jpg"), imagem) 


