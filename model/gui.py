import sys
import os
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QFileDialog,
)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt

import serial
import time

import numpy as np
import pandas as pd
import torch
import cv2
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torch.cuda
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.models as models
import matplotlib.pyplot as plt
from torch.nn import TripletMarginLoss
import seaborn as sns
from sklearn.metrics import confusion_matrix
import warnings

warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet18(pretrained=True)
model.fc = nn.Linear(512, 5)
model.load_state_dict(torch.load("saved models/512x5-98-96.pth", map_location=device))
model.to(device)
model.eval()

led_numbers = [8, 7, 6, 5, 4]
class_labels = ["Ben Afflek", "Elton John", "Jerry Seinfeld", "Madonna", "Mindy Kaling"]

store_last_k = 4


def delay(duration):
    ser = serial.Serial("COM18", baudrate=9600)
    start = time.time()
    while time.time() - start < duration:
        ser.write((chr(store_last_k + 65).lower()).encode())
        time.sleep(0.01)


# async def serial_write(ser):
#     start = time.time()
#     while time.time() - start < 2.133:
#         ser.write((chr(store_last_k + 65).lower()).encode())
#         # time.sleep(0.01)
#     return 1


def predict(model, image):
    def detect_and_crop_face(image):
        # Load the pre-trained face detection model
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        # Convert image to grayscale
        # gray = np.array(image) if len(image.shape) == 2 else cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)

        # Detect faces in the image
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )

        if len(faces) == 0:
            # No faces detected, return original image
            return image

        # Get the largest detected face
        (x, y, w, h) = max(faces, key=lambda face: face[2] * face[3])

        # Crop the face region from the image
        cropped_face = image.crop((x, y, x + w, y + h))

        return cropped_face

    # Preprocessing steps (same as your test_transform)
    preprocess = transforms.Compose(
        [
            transforms.Lambda(lambda img: detect_and_crop_face(img)),
            transforms.Resize((224, 224)),  # Resize images to 224x224
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize(mean=[0.5], std=[0.5]),  # Normalize
        ]
    )
    # Convert image to RGB and preprocess
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # image = Image.fromarray(image)
    image_tensor = preprocess(image)

    # Add an extra dimension for batch size (assumes single image inference)
    image_tensor = image_tensor.unsqueeze(0)

    # Set model to evaluation mode
    # model.eval()

    # Perform inference
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        output = model(image_tensor)

    # Get predicted class with highest probability (assuming classification task)
    _, predicted_class = torch.max(output.data, 1)

    # Replace 'class_labels' with your actual class labels
    # Example class labels
    predicted_label = class_labels[predicted_class.item()]

    return predicted_label


def load_and_predict(resnet, image_path):

    # Load image from file
    image = Image.open(image_path)

    # Predict on loaded image
    predicted_label = predict(resnet, image)
    print(f"The celebrity is {predicted_label}")

    return predicted_label


class ImageClassifierApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.uploaded_image_path = None

    def initUI(self):
        self.setWindowTitle("Face Recognition on FPGA")
        self.setGeometry(100, 100, 400, 300)

        self.image_label = QLabel("No image selected")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet(
            "QLabel { border: 2px dashed #aaaaaa; border-radius: 10px; background-color: #f0f0f0; }"
        )

        self.upload_button = QPushButton("Upload Image")
        self.upload_button.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; border: none; border-radius: 5px; padding: 8px 16px; font-size: 14px; } QPushButton:hover { background-color: #45a049; }"
        )

        self.classify_button = QPushButton("Classify")
        self.classify_button.setStyleSheet(
            "QPushButton { background-color: #008CBA; color: white; border: none; border-radius: 5px; padding: 8px 16px; font-size: 14px; } QPushButton:hover { background-color: #0071a4; }"
        )

        self.clear_button = QPushButton("Clear Image")
        self.clear_button.setStyleSheet(
            "QPushButton { background-color: #f44336; color: white; border: none; border-radius: 5px; padding: 8px 16px; font-size: 14px; } QPushButton:hover { background-color: #d32f2f; }"
        )

        button_layout = QVBoxLayout()
        button_layout.addWidget(self.upload_button)
        button_layout.addWidget(self.classify_button)
        button_layout.addWidget(self.clear_button)
        button_layout.addStretch()

        main_layout = QHBoxLayout()
        main_layout.addWidget(self.image_label)
        main_layout.addLayout(button_layout)
        self.setLayout(main_layout)

        self.upload_button.clicked.connect(self.uploadImage)
        self.classify_button.clicked.connect(self.classifyImage)
        self.clear_button.clicked.connect(self.clearImage)

    def uploadImage(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(
            self,
            options=options,
        )
        if fileName:
            pixmap = QPixmap(fileName)
            self.image_label.setPixmap(pixmap.scaledToWidth(300))
            self.uploaded_image_path = fileName

    def classifyImage(self):

        if self.uploaded_image_path:
            image_path = self.uploaded_image_path
            predicted_label = load_and_predict(model, image_path)

            if predicted_label is None:
                print("Failed to load image or perform prediction.")
        else:
            print("No image selected.")

    def clearImage(self):
        self.image_label.clear()
        self.image_label.setText("No image selected")
        self.uploaded_image_path = None


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageClassifierApp()
    window.show()
    sys.exit(app.exec_())
