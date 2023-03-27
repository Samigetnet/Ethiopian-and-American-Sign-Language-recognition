#import keras.saving.save
import numpy as np
from keras.models import load_model
def getLetter(result):
    classLabels ={0:'A',
                 1:'B',
                 2:'C',
                 3:'D',
                 4:'E',
                 5:'F',
                 6:'G',
                 7:'H',
                 8:'I',
                 9:'K',
                 10:'L',
                 11:'M',
                 12:'N',
                 13:'O',
                 14:'P',
                 15:'Q',
                 16:'R',
                 17:'S',
                 18:'T',
                 19:'U',
                 20:'V',
                 21:'W',
                 22:'X',
                 23:'Y', }
    try:
        res=int(result)
        return classLabels[res]
    except:
        return "Error"

import cv2
# cap= cv2.VideoCapture(0)
cap = cv2.VideoCapture(0)
model =load_model('sign_lang_recognition.h5')



while True:
    ret, frame = cap.read()

    roi = frame[100:400, 320:620]
    cv2.imshow('roi', roi)
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)

    cv2.imshow('roi sacled and gray', roi)
    copy = frame.copy()
    cv2.rectangle(copy, (320, 100), (620, 400), (255, 0, 0), 5)

    roi = roi.reshape(1, 28, 28, 1)
    print("ssssssssssss")
    pred = model.predict(roi)[0]
    max = str(np.argmax(pred))
    # result = str(model.predict_classes(roi, 1, verbose=0)[0])

    # cv2.putText(copy, getLetter(result), (300, 100), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 2)
    cv2.putText(copy, getLetter(max), (300, 100), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 2)
    cv2.imshow('frame', copy)

    if cv2.waitKey(1) == 13:
        break
cap.release()
cv2.destroyeAllWindows()