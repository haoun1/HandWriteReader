import os
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.optimizers as optims
import tensorflow.keras.models as models
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# 순서정리
# 1. 트레인셋을 작성한다 (x_train,y_train)
# 2. 모델의 형태(layer)를 설정한다.
# 3. 모델의 loss함수, optimizer를 설정한다 (컴파일작업) 
# 4. model.fit를 통해서 트레이닝을 시작한다. verbose는 상세 로그 출력여부를 결정한다. validation_split은 비율만큼을 testset으로 설정하며 이 set에 대해서는 가중치를 변경하지 않는다. loss는 val_loss로 따로 저장한다.#

def train(x_train, y_train, x_test, y_test):
    model = models.Sequential([
    layers.Dense(128, activation="relu", input_shape= x_train.shape[1:]),
    layers.Dense(y_train.shape[1], activation="softmax")
    ])
    
    model.compile(loss="categorical_crossentropy",
                  optimizer='adam',
                 metrics=['acc'] )
    model.summary()

    history = model.fit(x_train, y_train, batch_size=32, epochs=10, validation_split=0.1)
    loss, acc = model.evaluate(x_test, y_test)
    print("손실률:", loss)
    print("정확도", acc)

    plt.figure(figsize=(18,6))

    #에포크별 정확도
    plt.subplot(1,2,1)
    plt.plot(history.history["acc"], label = "accuracy")
    plt.plot(history.history["val_acc"], label = "val_accuracy")
    plt.title("accuracy")
    plt.legend()

    #에포크별 손실률
    plt.subplot(1,2,2)
    plt.plot(history.history["loss"], label="loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.title("loss")
    plt.legend()

    plt.show()

    model.save("model.h5")

    return model

def main():
    # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    # np.random.seed(0)
    # x_train = np.random.rand(650, 2)
    # coef = np.array([3,2])
    # bias = 10
    # y_train = np.matmul(x_train, coef.transpose()) + bias
    
    # # 선형 모델 선언, 
    # model = models.Sequential([
    #     layers.Input(2, name = 'input_layer'),
    #     layers.Dense(16, activation='sigmoid', name = 'hidden'),
    #     layers.Dense(1, activation='relu', name = 'output')
    # ])

    # # 모델의 loss_function과 optimizer를 설정한다.
    # model.compile(loss='mse', optimizer='sgd')

    # # train 시작
    # hist = model.fit(x_train,y_train,epochs=10, verbose=2, validation_split=0.3)

    # x = np.array([1,2])
    # x = x.reshape(1,2)
    # result = model.predict(x)

    # print(result)

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
    model = train(x_train, y_train, x_test, y_test)
    i = 9872

    plt.imshow(x_test[i].reshape(28,28))
    plt.show()
    predict_x=model.predict(x_test[i:i+1])
    classes_x=np.argmax(predict_x,axis=1)
    print("real:", y_test[i].argmax())
    print("predict:", predict_x.argmax())

if __name__ == '__main__':
    main()