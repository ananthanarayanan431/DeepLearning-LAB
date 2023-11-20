import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import to_categorical

df = load_digits().images
target = load_digits().target

X = df
y = target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=42)
w, h = df[0].shape

X_train = X_train.reshape(len(X_train), w * h)
X_test = X_test.reshape(len(X_test), w * h)
y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)


def Prediction_cat(model, image, value):
    pred = model.predict(image)
    print("Given Number: ", value, "Predicted Num: ", pred.argmax())


model = Sequential()

model.add(Dense(64, input_shape=X_train[0].shape, activation="relu"))
model.add(Dense(10, activation="softmax"))
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["acc"])

history = model.fit(X_train, y_train_cat, batch_size=32, epochs=20, validation_split=0.1)

result = model.evaluate(X_test.reshape(629, 64), y_test_cat)
for i in range(len(model.metrics_names)):
    print(model.metrics_names[i], ":", result[i])

for i in range(5):
    Prediction_cat(model, X_test[i].reshape(1, -1), y_test[i])