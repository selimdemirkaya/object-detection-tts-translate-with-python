# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 14:23:50 2019

@author: selim demirkaya
"""

from keras.applications.imagenet_utils import decode_predictions
from keras.applications import xception
import cv2 
#from matplotlib import pyplot as plt
import numpy as np
model = xception.Xception(weights='imagenet')
from keras.preprocessing import image
from keras.applications.xception import preprocess_input
import numpy as np
from matplotlib import pyplot as plt
from cv2 import *
import time
from gtts import gTTS 
import os 
from translate import Translator
from pygame import mixer

wrongvoice = 0

model = xception.Xception(weights='imagenet')
tts = 'tts'
for i in range(100):
    cam = VideoCapture(0)   # 0 -> index of camera
    s, img = cam.read()
    if s:    # frame captured without any errors.
        namedWindow("cam-test")
        imshow("cam-test",img)
        destroyWindow("cam-test")
        imwrite("filename.jpg",img)     
        
   
    img_path = 'filename.jpg'
    img = image.load_img(img_path, target_size=(299,299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    predict = model.predict(x)
    plt.figure(i)
    plt.imshow(img)
    plt.show()
    
    if decode_predictions (predict)[0][0][2] == "süzgeç" :    
        wrongvoice = 1
    if wrongvoice == 0 :
        if decode_predictions (predict)[0][0][2] >= 0.2 :
            mytext = decode_predictions(predict)[0][0][1]
            translator= Translator(to_lang="tr")
            mytext = translator.translate(mytext)
            print(decode_predictions(predict)[0])
            print(mytext)
            if decode_predictions (predict)[0][0][2] < 0.6 and decode_predictions (predict)[0][0][2] >= 0.2 and decode_predictions (predict)[0][0][2] < 0.3 :
                mytext = str ("galiba bu bir " + mytext)
                
            if decode_predictions (predict)[0][0][2] < 0.6 and decode_predictions (predict)[0][0][2] >= 0.3 and decode_predictions (predict)[0][0][2] < 0.4 :
                mytext = str ("sanırsam   bu bir" + mytext)
                
            if decode_predictions (predict)[0][0][2] < 0.6 and decode_predictions (predict)[0][0][2] >= 0.4 and decode_predictions (predict)[0][0][2] < 0.5 :
                mytext = str ("emin değilimm ama galiba bu bir " + mytext)
                
            if decode_predictions (predict)[0][0][2] < 0.6 and decode_predictions (predict)[0][0][2] >= 0.5 and decode_predictions (predict)[0][0][2] < 0.6 :
                mytext = str ("Bu bir " + mytext + "olabilir")
            
               
    language = 'tr'
    myobj = gTTS(text=mytext, lang=language, slow=False) 
    tts = gTTS(text=mytext, lang = 'tr', slow=False)
    file1 = str("" + str(i) + ".mp3")
    tts.save(file1)
    #os.remove(file1)
    
   
        
    print(file1)
    myobj.save(file1) 
    time.sleep(1)
    #os.system(file1)
    mixer.init()
    mixer.music.load(file1)
    mixer.music.play()
    
    #os.remove(file1)
    time.sleep(1)