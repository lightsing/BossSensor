# -*- coding:utf-8 -*-
import cv2
import numpy as np
from scipy.spatial import distance

from keras.engine import Model
from keras.layers import Input
from keras.preprocessing import image
from keras_vggface.vggface import VGGFace
from keras.applications.vgg19 import preprocess_input


class Recognizor:

    """Feature Extract Class."""

    def __init__(self):
        """Init."""
        self.model = VGGFace()
        self.out = self.model.get_layer('fc7').output
        self.model_fc7 = Model(self.model.input, self.out)

    def getFeature(self, img):
        """Get Feature From Face Image Object."""
        features = self.model_fc7.predict(img)
        return features


def loadImage(name):
    """Load Image From Disk."""
    img = image.load_img(name, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img


if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    cascade_path = "/usr/local/opt/opencv3/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(cascade_path)
    handler = Recognizor()
    bossImage = loadImage('data/boss.png')
    std = handler.getFeature(bossImage)
    while True:
        _, frame = cap.read()

        raw = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # グレースケール変換
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 物体認識（顔認識）の実行
        facerect = cascade.detectMultiScale(frame_gray, scaleFactor=1.2, minNeighbors=3, minSize=(10, 10))
        # facerect = cascade.detectMultiScale(frame_gray, 1.1, 20)
        if len(facerect) > 0:
            print('face detected')
            for (x, y, w, h) in facerect:
                img = cv2.resize(raw[y: y + h, x: x + w], (224, 224))
                img = np.expand_dims(img.astype(float), axis=0)
                img = preprocess_input(img)
                fn = handler.getFeature(img)
                d = distance.cosine(fn, std)
                if d < 0.1:
                    print('boss detected')

    # キャプチャを終了
    cap.release()
