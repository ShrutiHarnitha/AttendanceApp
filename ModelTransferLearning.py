import numpy as np
import pandas as pd
import csv
import tensorflow as tf
from sklearn.model_selection import train_test_split
import cv2
from pathlib import Path
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.optimizers import Adam
from keras.applications import vgg16

def ModelFineTuning():
    # Define the path to your dataset
    data_dir = Path('Dataset')
    image_size = (224, 224)  # VGGFace model expects 224x224 images

    # Initialize dictionaries
    candidates_dict = {}
    labels_dict = {}

    # Get all class folder names
    class_folders = [folder.name for folder in data_dir.iterdir() if folder.is_dir()]
    total_classes = len(class_folders)

    # Assign labels to each class
    for idx, class_name in enumerate(class_folders):
        candidates_dict[class_name] = list(data_dir.glob(f'{class_name}/*'))
        labels_dict[class_name] = idx

    df = pd.DataFrame(list(labels_dict.items()), columns=['Candidate Name', 'Label'])
    df.to_csv("candidate_labels.csv", index=False)

    # Print the results
    print('Images Dictionary:')
    print(candidates_dict)
    print('\nLabels Dictionary:')
    print(labels_dict)

    X, y = [], []
    if len(candidates_dict.items()) == 0:
        return False
    for candidate_name, faces in candidates_dict.items():
        for image in faces:
            img = cv2.imread(str(image))
            resized_img = cv2.resize(img, image_size)
            X.append(resized_img)
            y.append(labels_dict[candidate_name])

    print(len(X))

    X = np.array(X)
    y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    X_train_scaled = X_train / 255.0
    X_test_scaled = X_test / 255.0

    # Convert labels to one-hot encoding
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=total_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=total_classes)

    # Load the pre-trained VGGFace model
    base_model = vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # Ensure the base model layers are not trainable
    for layer in base_model.layers:
        layer.trainable = False

    # Create a Sequential model and add layers
    model = Sequential()
    model.add(Input(shape=(224, 224, 3)))
    model.add(base_model)
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(total_classes, activation='softmax'))

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    history = model.fit(
        X_train_scaled, y_train,
        validation_data=(X_test_scaled, y_test),
        epochs=10,  # Adjust the number of epochs based on your needs
        batch_size=32
    )

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test_scaled, y_test)
    print(f"Test accuracy: {accuracy * 100:.2f}%")

    # Save the fine-tuned model
    model.save('fine_tuned_VGG16_model.h5')
    return True

# ModelFineTuning()  # Uncomment this line to run the training
