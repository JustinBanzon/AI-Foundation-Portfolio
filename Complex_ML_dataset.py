import matplotlib.pyplot as plt
import numpy as np
import nnfs
from nnfs.datasets import spiral_data
nnfs.init()

class LayerDense:
    def __init__(self,n_inputs,n_neurons):
        self.weights = np.random.randn(n_inputs,n_neurons) * np.sqrt(2 / n_inputs)
        self.biases = np.zeros((1,n_neurons))
    def forward(self,inputs):
        self.output = np.dot(inputs,self.weights) + self.biases
relu = lambda inputs:np.maximum(0,inputs)
class ActivationReLU:
    def forward(self, inputs):
        self.output = relu(inputs)
def softmax(inputs):
    exp_values = np.exp(inputs - inputs.max(axis=1,keepdims=True))
    probabilities = exp_values/exp_values.sum(axis = 1, keepdims = True)
    return probabilities
class ActivationSoftmax:
    def forward(self,inputs):
        self.output = softmax(inputs)
class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output,y)
        data_loss = np.mean(sample_losses)
        return data_loss
class Loss_CatecoricalCrossEntropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred,1e-15,1-1e-15)
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples),y_true]
        elif len(y_true.shape == 2):
            correct_confidences = np.sum(y_pred_clipped*y_true, axis=1)
        negative_log_likeliehoods = -np.log(correct_confidences)
        return negative_log_likeliehoods

X, y = spiral_data(samples=450,classes=3)
D = X.shape[1]
H = 32

dense_in = LayerDense(D,H)
activation_in = ActivationReLU()
dense_hidden1 = LayerDense(H,H)
activation_hidden1 = ActivationReLU()
dense_hidden2 = LayerDense(H,H)
activation_hidden2 = ActivationReLU()
dense_out = LayerDense(H,3)
activation_out = ActivationSoftmax()

print(dense_in.weights.shape,dense_in.biases.shape,dense_out.weights.shape,dense_out.biases.shape)
loss_function = Loss_CatecoricalCrossEntropy()

lr = 0.05
epochs = 500
batch = 64
losses = []
accuracies = []
for _ in range(epochs):
    idx = np.random.permutation(len(X))
    Xs, Ys = X[idx], y[idx]
    for i in range(0, len(Xs), batch):
        xb = Xs[i:i + batch]
        yb = Ys[i:i + batch]
        z1 = xb.dot(dense_in.weights) + dense_in.biases
        a1 = relu(z1)
        z2 = a1.dot(dense_hidden1.weights) + dense_hidden1.biases
        a2 = relu(z2)
        z3 = a2.dot(dense_hidden2.weights) + dense_hidden2.biases
        a3 = relu(z3)
        logits = a3.dot(dense_out.weights) + dense_out.biases
        probs = softmax(logits)
        Yb = np.zeros_like(probs)
        Yb[np.arange(len(yb)), yb] = 1
        dlog = (probs - Yb) / len(yb)
        dW4 = a3.T.dot(dlog)
        db4 = dlog.sum(axis=0)
        da3 = dlog.dot(dense_out.weights.T)
        dz3 = da3 * (z3 > 0)
        dW3 = a2.T.dot(dz3)
        db3 = dz3.sum(axis=0)
        da2 = da3.dot(dense_hidden2.weights.T)
        dz2 = da2 * (z2 > 0)
        dW2 = a1.T.dot(dz2)
        db2 = dz2.sum(axis=0)
        da1 = da2.dot(dense_hidden1.weights.T)
        dz1 = da1 * (z1 > 0)
        dW1 = xb.T.dot(dz1)
        db1 = dz1.sum(axis=0)
        dense_out.weights -= lr * dW4
        dense_out.biases -= lr * db4
        dense_hidden2.weights -= lr * dW3
        dense_hidden2.biases -= lr * db3
        dense_hidden1.weights -= lr * dW2
        dense_hidden1.biases -= lr * db2
        dense_in.weights -= lr * dW1
        dense_in.biases -= lr * db1

    dense_in.forward(X)
    activation_in.forward(dense_in.output)
    dense_hidden1.forward(activation_in.output)
    activation_hidden1.forward(dense_hidden1.output)
    dense_hidden2.forward(activation_hidden1.output)
    activation_hidden2.forward(dense_hidden2.output)
    dense_out.forward(activation_hidden2.output)
    activation_out.forward(dense_out.output)
    
    loss = loss_function.calculate(activation_out.output,y)
    losses.append(loss)
    
    predictions = np.argmax(activation_out.output, axis=1)
    accuracy = np.mean(predictions==y)
    accuracies.append(accuracy)

plt.figure()
plt.plot(losses, color="red")
plt.title("Neural Network — Loss")
plt.xlabel("Epoch")
plt.grid(True)
plt.show()

plt.figure()
plt.plot(accuracies, color="green")
plt.title("Neural Network — Accuracy")
plt.xlabel("Epoch")
plt.grid(True)
plt.show()

test_X, test_y = spiral_data(samples=450,classes=3)
xx, yy = np.meshgrid(
    np.linspace(test_X[:, 0].min() - 1, test_X[:, 0].max() + 1, 300),
    np.linspace(test_X[:, 1].min() - 1, test_X[:, 1].max() + 1, 300)
)
grid = np.c_[xx.ravel(), yy.ravel()]
dense_in.forward(grid)
activation_in.forward(dense_in.output)
dense_hidden1.forward(activation_in.output)
activation_hidden1.forward(dense_hidden1.output)
dense_hidden2.forward(activation_hidden1.output)
activation_hidden2.forward(dense_hidden2.output)
dense_out.forward(activation_hidden2.output)
activation_out.forward(dense_out.output)
activation_array = np.array(activation_out.output)
pred = np.argmax(activation_array,axis=1).reshape(xx.shape)
plt.figure()
plt.contourf(xx, yy, pred, 3)
plt.scatter(test_X[:, 0], test_X[:, 1], c=test_y, edgecolor="black")
plt.title("Neural Network — Decision Regions")
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()