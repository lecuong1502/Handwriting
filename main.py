import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# mnist = tf.keras.datasets.mnist
# (x_train, y_train), (x_test, y_test) = mnist.load_data() #Preprocess the data

# x_train = tf.keras.utils.normalize(x_train, axis=1)   #Normalize the data 0-255
# x_test = tf.keras.utils.normalize(x_test, axis=1)

# # Build the neural network model
# model = tf.keras.models.Sequential()   #Model is a sequence of layers
# model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))   #Dataset is 28x28 pixels
# model.add(tf.keras.layers.Dense(128, activation='relu'))  #Dense layer with 128 neurons and ReLU activation
# model.add(tf.keras.layers.Dense(128, activation='relu'))  #Another dense layer with 128 neurons
# model.add(tf.keras.layers.Dense(10, activation='softmax'))  #Output layer with 10 neurons for 10 classes (digits 0-9)

# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])  #Compile the model with Adam optimizer and sparse categorical crossentropy loss
# model.fit(x_train, y_train, epochs=5) # x_train is the training data, y_train is the labels, epochs is the number of times to train on the dataset
# model.save('mhandwriting.keras')


model = tf.keras.models.load_model('mhandwriting.keras')  #Load the trained model

# loss, accuracy = model.evaluate(x_test, y_test)  #Evaluate the model on the test data
# print(f"Loss: {loss}, Accuracy: {accuracy}")  #Print the loss and accuracy of the model

image_number = 3
while os.path.isfile(f"digits/digit{image_number}.png"):
    try:
        img_original = cv2.imread(f"digits/digit{image_number}.png", cv2.IMREAD_GRAYSCALE)

        if img_original is None:
            print(f"Warning: Could not read image {image_number}. Skipping.")
            image_number += 1
            continue

        img_resized = cv2.resize(img_original, (28, 28), interpolation=cv2.INTER_AREA)

        img_inverted = np.invert(img_resized)

        img_normalized = tf.keras.utils.normalize(img_inverted, axis=1)

        img_for_prediction = np.array([img_normalized])

        prediction = model.predict(img_for_prediction)
        print(f"Prediction for image {image_number}: {np.argmax(prediction)}")

        plt.imshow(img_inverted, cmap=plt.cm.binary)
        plt.title(f"Predicted: {np.argmax(prediction)}")
        plt.show()

    except Exception as e:
        print(f"Error processing image {image_number}: {e}")
    finally:
        image_number += 1
print("Processing complete.")
plt.close('all')
cv2.destroyAllWindows()
tf.keras.backend.clear_session()
print("Session cleared.")