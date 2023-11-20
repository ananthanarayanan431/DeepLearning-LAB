import numpy as np

input_value = np.array([[0,0],[0,1],[1,1],[1,0]])
print("Input_Array shape\n")
print(input_value.shape)

print("\nInput Array")
print(input_value)

output_value = np.array([0,1,1,0])
output_value = output_value.reshape(4,1)

print("\nOutput Array Shape\n")
print(output_value.shape)

weights = np.array([[0.1],[0.2]])
print("\nWeights\n")
print(weights)

bias = 0.3

def sigmoid_func(x):
    return 1/(1+np.exp(-x))

def derv(x):
    return sigmoid_func(x) * (1-sigmoid_func(x))


for epochs in range(1000):
    input_arr = input_value

    weighted_sum = np.dot(input_arr, weights) + bias
    first_output = sigmoid_func(weighted_sum)

    error = first_output - output_value
    total_wrror = np.square(np.subtract(first_output, output_value)).mean()

    first_der = error
    second_der = derv(first_output)
    derivative = first_der * second_der

    t_input = input_value.T
    final_derivative = np.dot(t_input, derivative)

    weights = weights - 0.05 * final_derivative

    for i in derivative:
        bias = bias - 0.05 * i

    # print(epochs,weights,end=" ")
    # print(epochs,bias,end=" ")
print("\nUpdated Weights and Bias\n")
print(weights)
print(bias)

print("\n\n")
pred = np.array([0,1])

result = np.dot(pred,weights) + bias

res = sigmoid_func(result)

print("\nFinal Output\n")
print(res)


# After adding changes on 20/11/2023 for redcuing the no lines in the code

# from keras.models import Sequential
# from keras.layers import Dense
# import numpy as np

# input_value = np.array([[0, 0], [0, 1], [1, 1], [1, 0]])
# output_value = np.array([0, 1, 1, 0])

# model = Sequential()
# model.add(Dense(100, input_dim=2, activation='relu'))
# model.add(Dense(50, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))

# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# model.fit(input_value, output_value, epochs=1000, batch_size=4, verbose=0)

# # Evaluate the model
# loss, accuracy = model.evaluate(input_value, output_value, verbose=0)
# print(f"Accuracy: {accuracy}")

# # Prediction
# pred = np.array([[0, 1]])  # Predict for input [0, 1]
# result = model.predict(pred)
# print("\nFinal Output\n")
# print(result)
