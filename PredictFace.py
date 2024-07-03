import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
from pathlib import Path
from datetime import datetime
from tensorflow.keras.models import load_model


# Function to preprocess the image
def preprocess_image(image_path, target_size=(224, 224)):
    img = cv2.imread(image_path)
    img = cv2.resize(img, target_size)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Function to predict the label of the image
def predict_candidate(image_path):

    model = load_model('fine_tuned_VGG16_model.h5')

    df = pd.read_csv('candidate_labels.csv')
    labels_dict = df.set_index('Label')['Candidate Name'].to_dict()

    if image_path == '':
        image_path = 'Dataset/test_img.jpg' 

    print("Image Path is", image_path)

    img = preprocess_image(image_path)
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction, axis=1)[0]
    predicted_label = labels_dict[predicted_class]

    # Get the current timestamp in 12-hour format without seconds
    timestamp = datetime.now().strftime('%Y-%m-%d %I:%M %p')
    # Save to CSV
    attendance_record = pd.DataFrame([[timestamp, predicted_label]], columns=['Timestamp', 'Student ID Number'])
    attendance_record.to_csv('Attendance_Record.csv', mode='a', header=False, index=False)

    print("Label is", predicted_label)
    return predicted_label

