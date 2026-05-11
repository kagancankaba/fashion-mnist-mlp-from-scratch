import numpy as np
import tensorflow as tf


(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()


train_images = train_images.reshape(60000, 784) / 255.0
test_images = test_images.reshape(10000, 784) / 255.0


train_labels = np.eye(10)[train_labels]
test_labels = np.eye(10)[test_labels]


def initialize_parameters(input_size, hidden_size, output_size):
    np.random.seed(42)

    W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2. / input_size)
    b1 = np.zeros(hidden_size)
    W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2. / hidden_size)#0.01 ile carpialrak da uygulanabilir
    b2 = np.zeros(output_size)
    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}

parameters = initialize_parameters(784, 128, 10)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def forward_prop(X, parameters):
   
    Z1 = np.dot(X, parameters["W1"]) + parameters["b1"]
    A1 = sigmoid(Z1)
    
    Z2 = np.dot(A1, parameters["W2"]) + parameters["b2"]
    A2 = sigmoid(Z2)
    
    return {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}

def compute_loss(A2, Y):
    m = Y.shape[0]
    loss = -np.sum(Y * np.log(A2 + 1e-8)) / m  
    return loss

def back_prop(parameters, cache, X, Y):
    m = X.shape[0]
    
    dZ2 = cache["A2"] - Y  
    dW2 = np.dot(cache["A1"].T, dZ2) / m
    db2 = np.sum(dZ2, axis=0) / m
    
    dA1 = np.dot(dZ2, parameters["W2"].T)
    dZ1 = dA1 * sigmoid_derivative(cache["Z1"])
    dW1 = np.dot(X.T, dZ1) / m
    db1 = np.sum(dZ1, axis=0) / m
    
    return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}

def update_parameters(parameters, grads, learning_rate=0.1):
    parameters["W1"] -= learning_rate * grads["dW1"]
    parameters["b1"] -= learning_rate * grads["db1"]
    parameters["W2"] -= learning_rate * grads["dW2"]
    parameters["b2"] -= learning_rate * grads["db2"]
    return parameters

batch_size = 64
epochs = 20

for epoch in range(epochs):
    for i in range(0, train_images.shape[0], batch_size):
        
        X_batch = train_images[i:i+batch_size]
        Y_batch = train_labels[i:i+batch_size]
        
        cache = forward_prop(X_batch, parameters)
        
        loss = compute_loss(cache["A2"], Y_batch)
        
        grads = back_prop(parameters, cache, X_batch, Y_batch)
        
        parameters = update_parameters(parameters, grads, 0.1)
    
    print(f"Epoch {epoch+1}, Loss: {loss:.4f}")
    
cache_test = forward_prop(test_images, parameters)
predictions = np.argmax(cache_test["A2"], axis=1)
true_labels = np.argmax(test_labels, axis=1)
accuracy = np.mean(predictions == true_labels)
print(f"Test Doğruluğu: {accuracy * 100:.2f}%")

import matplotlib.pyplot as plt

class_names = ['Tişört/Üst', 'Pantolon', 'Kazak', 'Elbise', 'Ceket',
               'Sandalet', 'Gömlek', 'Spor Ayakkabı', 'Çanta', 'Bilekte Bot']

last_15_images = test_images[-15:].reshape(15, 28, 28)
last_15_labels = true_labels[-15:]
last_15_predictions = predictions[-15:]

plt.figure(figsize=(15, 9))
for i in range(15):
    plt.subplot(3, 5, i+1)  
    plt.imshow(last_15_images[i], cmap='gray')
    
    pred_name = class_names[last_15_predictions[i]]
    true_name = class_names[last_15_labels[i]]

    color = 'green' if last_15_predictions[i] == last_15_labels[i] else 'red'
    plt.title(f"T: {pred_name[:10]}...\nG: {true_name}",
             color=color, 
             fontsize=9, 
             pad=2)
    plt.axis('off')

plt.tight_layout()
plt.show()

print("\nSon 15 Tahmin Sonucu:")
for i in range(15):
    print(f"Resim {i+1:2}: {'✅' if last_15_predictions[i] == last_15_labels[i] else '❌'} {class_names[last_15_labels[i]]} → {class_names[last_15_predictions[i]]}")