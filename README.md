# Handwritten Digit Recognition

## Project Overview

This project is a web application that recognizes handwritten digits using a Convolutional Neural Network (CNN) trained on the MNIST dataset. Users can either draw a digit on the canvas or upload an image, and the trained model predicts the digit with a confidence score.

---

## Features

- Draw handwritten digits using the mouse.
- Upload digit images.
- Automatic preprocessing.
- CNN-based digit recognition.
- Prediction confidence.
- Light/Dark mode.
- Responsive interface.

---

## Technologies Used

### Backend

- Python
- Flask

### Machine Learning

- TensorFlow
- Keras

### Frontend

- HTML5
- CSS3
- JavaScript

### Libraries

- NumPy
- Pillow
- SciPy

---

## Project Structure

```
project/

│

├── app.py

├── train_model.py

├── digit_model.keras

├── templates/

│ └── index.html

├── static/

│ ├── script.js

│ └── style.css

└── README.md
```

---

---

## Machine Learning Model

The model is implemented using a Convolutional Neural Network (CNN) trained on the MNIST handwritten digit dataset. Data augmentation techniques such as rotation, zooming, and translation are applied during training to improve the model's robustness.

---

## Workflow

1. User draws or uploads an image.
2. JavaScript sends the image to Flask.
3. Flask preprocesses the image.
4. CNN predicts the digit.
5. Flask returns the prediction.
6. JavaScript displays the predicted digit and confidence score.

---

## Installation

Install the required libraries:

````bash
pip install -r requirements.txt


Run the Project

Run the Flask server:
```bash

python app.py

Open the browser:

```bash

http://127.0.0.1:7860

Team Members
Khaled_290137
batoul_341595
ghalia_371127
Nour_394817
mohammad_302362
...
...
Future Improvements
Support multiple digit recognition.
Improve model accuracy.
Mobile optimization.
Deploy the application online.

License

This project was developed for educational purposes.
````
