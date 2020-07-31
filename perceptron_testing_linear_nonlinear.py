import numpy as np
from perceptron_linear_nonlinear import Perceptron

training_inputs = []
training_inputs.append(np.array([0,0,0,1]))
training_inputs.append(np.array([0,0,1,0]))
training_inputs.append(np.array([0,0,1,1]))
training_inputs.append(np.array([0,1,0,0]))
training_inputs.append(np.array([0,1,0,1]))
training_inputs.append(np.array([0,1,1,0]))
training_inputs.append(np.array([0,1,1,1]))
training_inputs.append(np.array([1,0,0,0]))
training_inputs.append(np.array([1,0,0,1]))
training_inputs.append(np.array([1,0,1,0]))
training_inputs.append(np.array([1,0,1,1]))
training_inputs.append(np.array([1,1,0,0]))
training_inputs.append(np.array([1,1,0,1]))
training_inputs.append(np.array([1,1,1,0]))
training_inputs.append(np.array([1,1,1,1]))

labels = np.array([1,0,0,1,1,1,0,1,0,0,0,1,0,1,0,1])

perceptron = Perceptron(4)
perceptron.train(training_inputs, labels)

inputs = np.array([0,0,0,0])
print(*inputs, sep = ", ")
print("Expected Output is 1 and Actual Output is ", perceptron.predict(inputs))

inputs = np.array([1,1,1,1])
print(*inputs, sep = ", ")
print("Expected Output is 1 and Actual Output is ", perceptron.predict(inputs))

inputs = np.array([1,0,0,0])
print(*inputs, sep = ", ")
print("Expected Output is 0 and Actual Output is ", perceptron.predict(inputs))

inputs = np.array([0,1,0,1])
print(*inputs, sep = ", ")
print("Expected Output is 1 and Actual Output is ", perceptron.predict(inputs))