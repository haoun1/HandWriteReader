import os
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.optimizers as optims
import tensorflow.keras.models as models
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

def Train(batch, epoch):
    # 트레인셋 = 60000개, 테스트셋 = 1만개  x = 이미지, y = 그 이미지가 어떤 숫자인지
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    input_shape = x_train.shape[1] * x_train.shape[2] #그림의 크기 28 * 28 = 784
    number_of_classes = len(set(y_train)) #레이블의 종류 0~9 갯수 = 10개

    # MNIST는 0~255의 gv가 모여있는 데이터셋인데, 이를 0~1사이의 데이터셋으로 변경함
    # 또한, 60000,28,28의 데이터셋을 60000,784 인 1차원 데이터로 변경함
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    x_train = x_train.reshape(-1, input_shape)
    x_test = x_test.reshape(-1, input_shape)

    # 원 핫 인코딩 (어떤 숫자를 표현할 때 2 = {0,1,0,0,0,0,0,0,0,0} 5 = {0,0,0,0,0,1,0,0,0,0}) 이런식으로 표현하게 만드는방법
    y_train = to_categorical(y_train, number_of_classes) # (60000,) -> (60000, 10)
    y_test = to_categorical(y_test, number_of_classes)

    model = models.Sequential([
    layers.Dense(128, activation="relu", input_shape= x_train.shape[1:]),
    layers.Dense(y_train.shape[1], activation="softmax")
    ])
    
    model.compile(loss="categorical_crossentropy",
                  optimizer='adam',
                 metrics=['acc'] )
    model.summary()

    history = model.fit(x_train, y_train, batch_size = batch, epochs = epoch, validation_split=0.1)
    loss, acc = model.evaluate(x_test, y_test)
    print("손실률:", loss)
    print("정확도", acc)

    model.save("model.h5")

    return model
