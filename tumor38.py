import streamlit as st
import tensorflow as tf
import tempfile
import os
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score

# Define constants
IMG_HEIGHT = 224
IMG_WIDTH = 224
CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']
NUM_CLASSES = len(CLASS_NAMES)

# Title of your Streamlit app
st.title("Brain Tumor Classification")

# Upload the model file to Streamlit
model_file = st.file_uploader("Upload your model file (CNN_image_classifier.h5)", type=['h5'])

@st.cache_resource
def load_model(model_file):
    if model_file is not None:
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
            tmp.write(model_file.read())
            tmp.close()
            model = tf.keras.models.load_model(tmp.name)
            os.remove(tmp.name)
            return model
    else:
        return None

# Load the model
model = load_model(model_file)

# Check if the model is loaded
if model is not None:
    st.success("Model loaded successfully!")
else:
    st.error("Please upload your model file first.")

def predict_image(image):
    img = image.convert('RGB')
    img = img.resize((IMG_HEIGHT, IMG_WIDTH))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    confidence_score = np.max(prediction)

    return CLASS_NAMES[predicted_class], confidence_score, prediction

def load_and_preprocess_data(data_dir):
    images = []
    labels = []
    label_dict = {name: idx for idx, name in enumerate(CLASS_NAMES)}

    for label_name in CLASS_NAMES:
        label_dir = os.path.join(data_dir, label_name)
        for file_name in os.listdir(label_dir):
            file_path = os.path.join(label_dir, file_name)
            img = load_img(file_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
            img_array = img_to_array(img) / 255.0
            images.append(img_array)
            labels.append(label_dict[label_name])

    return np.array(images), np.array(labels)

def plot_confusion_matrix(y_true, y_pred_class):
    cm = confusion_matrix(y_true, y_pred_class)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, cmap="Blues", fmt='g', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    st.pyplot(plt)

def plot_roc_curve(y_true, y_score):
    plt.figure(figsize=(10, 8))
    roc_auc = {}
    for i in range(NUM_CLASSES):
        fpr, tpr, _ = roc_curve(y_true == i, y_score[:, i])
        roc_auc[i] = roc_auc_score(y_true == i, y_score[:, i])
        plt.plot(fpr, tpr, label=f'Class {CLASS_NAMES[i]} (AUC = {roc_auc[i]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    st.pyplot(plt)

    return roc_auc

def display_results(uploaded_file):
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        predicted_class_name, confidence_score, prediction = predict_image(image)
        st.sidebar.text(f'Predicted Class: {predicted_class_name}')
        st.sidebar.text(f'Confidence Score: {confidence_score:.4f}')

        st.session_state.results['prediction'] = (predicted_class_name, confidence_score)

def evaluate_model(test_data_dir):
    test_data, test_labels = load_and_preprocess_data(test_data_dir)
    y_pred = model.predict(test_data)
    y_pred_class = np.argmax(y_pred, axis=1)
    y_true = test_labels if len(test_labels.shape) == 1 else np.argmax(test_labels, axis=1)

    return y_true, y_pred, y_pred_class

def app():
    if 'results' not in st.session_state:
        st.session_state.results = {}

    st.sidebar.header("Upload Image")
    uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    display_results(uploaded_file)

    if st.sidebar.button('Show Classification Report'):
        y_true, _, y_pred_class = evaluate_model('Testing')
        st.session_state.results['classification_report'] = classification_report(y_true, y_pred_class, target_names=CLASS_NAMES)
        st.write("Classification Report:")
        st.text(st.session_state.results['classification_report'])

    if st.sidebar.button('Show Confusion Matrix'):
        y_true, _, y_pred_class = evaluate_model('Testing')
        st.session_state.results['confusion_matrix'] = (y_true, y_pred_class)
        st.write("Confusion Matrix:")
        plot_confusion_matrix(y_true, y_pred_class)

    if st.sidebar.button('Show Test Accuracy'):
        y_true, _, y_pred_class = evaluate_model('Testing')
        test_accuracy = np.mean(y_pred_class == y_true)
        st.session_state.results['test_accuracy'] = test_accuracy
        st.write(f"Test Accuracy: {test_accuracy:.4f}")

    if st.sidebar.button('Show ROC Curve'):
        y_true, y_pred, _ = evaluate_model('Testing')
        roc_auc = plot_roc_curve(y_true, y_pred)
        st.session_state.results['roc_auc'] = roc_auc

        st.write("ROC AUC Scores:")
        for i, score in roc_auc.items():
            st.write(f"Class {CLASS_NAMES[i]}: {score:.2f}")

if __name__ == "__main__":
    app()
