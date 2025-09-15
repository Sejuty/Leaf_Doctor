# Leaf Doctor 🌿: Plant Disease Detection App

[](https://leafdoctor.streamlit.app/)
[](https://www.python.org/)
[](https://opensource.org/licenses/MIT)

A user-friendly web application built to help you identify diseases in your Pepper, Potato, and Tomato plants. Simply upload an image of a leaf, and the deep learning model will predict the disease.

**[\>\> Visit the Live Application \<\<](https://leafdoctor.streamlit.app/)**

## 📋 About The Project

Leaf Doctor uses a powerful deep learning model (MobileNetV2) to classify 15 different types of plant diseases from an uploaded leaf image. The goal is to provide a simple, accessible tool for gardeners and farmers to quickly diagnose potential issues with their plants.

This application was trained on the PlantVillage dataset and is deployed for free using Streamlit Community Cloud.

## ✨ Features

  * **Image Upload**: Supports JPG, JPEG, and PNG image formats.
  * **Real-Time Prediction**: Get a diagnosis in seconds.
  * **Disease Identification**: Classifies among 15 common diseases for Pepper, Potato, and Tomato plants.
  * **Confidence Score**: Displays the model's confidence in its prediction.
  * **Responsive UI**: Clean and simple interface that works on desktop and mobile.

## 🛠️ Technologies Used

  * **Frontend**: Streamlit
  * **Backend & Model**: Python, TensorFlow, Keras, OpenCV
  * **Deployment**: Streamlit Community Cloud

## 🚀 How To Run Locally

To get a local copy up and running, follow these simple steps.

### Prerequisites

You need to have Python 3.8+ and pip installed on your system.

### Installation

1.  **Clone the repository:**

    ```sh
    git clone https://github.com/Sejuty/Leaf_Doctor.git
    ```

2.  **Navigate to the project directory:**

    ```sh
    cd Leaf_Doctor
    ```

3.  **Install the required packages:**

    ```sh
    pip install -r requirements.txt
    ```

    *Note: Make sure you have the model file (`plant_disease_model_15.h5`) in the same directory.*

4.  **Run the Streamlit app:**

    ```sh
    streamlit run app.py
    ```

    Your app should now be running locally at `http://localhost:8501`.

## 🧠 Model Information

  * **Architecture**: MobileNetV2 (using transfer learning)
  * **Dataset**: [PlantVillage Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease) from Kaggle.
  * **Classes**: 15 distinct classes.
  * **Validation Accuracy**: Achieved a validation accuracy of **90.51%** after 10 epochs of training.

## 📄 License

Distributed under the MIT License. See `LICENSE` file for more information.
