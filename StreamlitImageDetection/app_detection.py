import os
import pandas as pd
from PIL import Image
import numpy as np
import random
import seaborn as sns
from matplotlib import pyplot as plt
import os
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf
from tensorflow import keras
import streamlit as st
from keras.models import load_model

def main():
    # Set the page configuration
    st.title("Skin cancer diseases detection")
    st.write("This recognition app was designed from analysis of the Kaggle Data set https://www.kaggle.com/datasets/farjanakabirsamanta/skin-cancer-dataset.")
    st.write("Dataset included a HAM10000.csv file with information of 10015 skin lesions pictures provided.")
    st.write("To build the app with train models, different analysis were conducted and are explained on the different pages of this app.")
    st.write("- **Page 2:** EDA, datavisualization on HAM10000.csv ")
    st.write("- **Page 3:** ML analysis on HAM10000.csv")
    st.write("- **Page 4:** DL and ML analysis on HAM10000 pictures")
    st.write("- **Page 5:** NLP analysis from diseases text description")  
    ###Model Importing
    ###function for models
    ##LINEAR MODEL
    with open('/home/annemocoeur/STREAMLIT_APP/StreamlitImageDetection/model_rf_csv_analysis.sav', 'rb') as file:
        rf_model_csv = pickle.load(file)

    # Define a function to preprocess user input
    def preprocess_user_input(user_input):
        # Encode binary features (checkboxes) as 0 or 1
        user_input['follow_up'] = user_input['follow_up'].astype(int)
        user_input['confocal'] = user_input['confocal'].astype(int)
        user_input['consensus'] = user_input['consensus'].astype(int)
        user_input['histo'] = user_input['histo'].astype(int)

        # Ensure that columns are in the desired order
        desired_column_order = [
            'confocal', 'consensus', 'follow_up', 'histo',
            'abdomen', 'acral', 'back', 'chest', 'ear', 'face', 'foot', 'genital', 'hand',
            'lower extremity', 'neck', 'scalp', 'trunk', 'unknown', 'upper extremity',
            'age', 'sex'
        ]
    
        # Reorder columns
        user_input = user_input[desired_column_order]

        # Return the preprocessed user input
        return user_input

    # Create a Streamlit app
    st.header("Tell us more...")

    # Collect user input
    sex = st.radio("**You are**", ("Male", "Female"))
    age = st.slider("**Age**", min_value=20, max_value=100)
    localization = st.selectbox("**Where is the lesion located?**", ("abdomen", "acral", "back", "chest", "ear", "face", "foot", "genital", "hand", "lower extremity", "neck", "scalp", "trunk", "unknown", "upper extremity"))

    # Custom dropdown for Method of Detection
    method_of_detection = st.selectbox("**Method of Detection**", ("Follow-up", "Confocal", "Consensus", "Histo"))

    # Encode 'sex' as 1 for 'Male' and 0 for 'Female'
    sex_encoded = 1 if sex == "Male" else 0

    # Create a user input DataFrame
    user_input = pd.DataFrame({
        'confocal': [1 if method_of_detection == "Confocal" else 0],
        'consensus': [1 if method_of_detection == "Consensus" else 0],
        'follow_up': [1 if method_of_detection == "Follow-up" else 0],
        'histo': [1 if method_of_detection == "Histo" else 0],
        'abdomen': [1 if localization == "abdomen" else 0],
        'acral': [1 if localization == "acral" else 0],
        'back': [1 if localization == "back" else 0],
        'chest': [1 if localization == "chest" else 0],
        'ear': [1 if localization == "ear" else 0],
        'face': [1 if localization == "face" else 0],
        'foot': [1 if localization == "foot" else 0],
        'genital': [1 if localization == "genital" else 0],
        'hand': [1 if localization == "hand" else 0],
        'lower extremity': [1 if localization == "lower extremity" else 0],
        'neck': [1 if localization == "neck" else 0],
        'scalp': [1 if localization == "scalp" else 0],
        'trunk': [1 if localization == "trunk" else 0],
        'unknown': [1 if localization == "unknown" else 0],
        'upper extremity': [1 if localization == "upper extremity" else 0],
        'age': [age],
        'sex': [sex_encoded]
    })


    cnn_model_labels = ['Actinic keratoses / intraepithelial carcinoma, also named Bowens disease ', 'basal cell carcinoma', 'benign keratosis-like lesions (solar lentigines / seborrheic keratoses and lichen-planus like keratoses)', 'dermatofibroma', 'melanoma ', 'melanocytic nevi', 'vascular lesions (angiomas, angiokeratomas, pyogenic granulomas and hemorrhage)']

    # Preprocess user input
    preprocessed_input = preprocess_user_input(user_input)

    # Make a prediction using the model
    prediction_rf_csv = rf_model_csv.predict(preprocessed_input)
    rf_pred_csv = cnn_model_labels[prediction_rf_csv[0]]



    # Display the predictions
    st.write(f"Based on the information provided, it is possible that you may have a **{rf_pred_csv}**.")
    st.markdown("*This diagnose is predicted from a random forest model trained on over 10k information dataset.*")
    st.write("**For optimal predictions, please upload an image of the skin lesion in the next section.**")
    ###########UPLOAD IMAGE
    st.header("Select your picture")

    st.write("Provide a colored picture of your skin lesion centered.")
    # SCRIPT FOR MODEL PREDICTION
    ##Upload model
    model1 = tf.keras.models.load_model('/home/annemocoeur/STREAMLIT_APP/StreamlitImageDetection/cnn_mode1.h5')
    model1_bal_train = tf.keras.models.load_model('/home/annemocoeur/STREAMLIT_APP/StreamlitImageDetection/cnn_model1_bal_train.h5')
    ##LINEAR MODEL
    with open('/home/annemocoeur/STREAMLIT_APP/StreamlitImageDetection/model_rf_Image_Analysis.pkl', 'rb') as file:
        rf_model = pickle.load(file)

    ##Preprocessing images for CNN & ML models
    def preprocess_image_cnn(image_path):
        # Load the image using OpenCV
        image = image_path.resize((28, 28))
        image = np.array(image)
        image = image / 255.0 
        image_flattened = image.reshape(1, -1)
        return image, image_flattened

    ###########Diseases descriptio
    uploaded_image  = st.file_uploader("Upload picture", type=["jpg", "png", "jpeg"])

    if uploaded_image is not None:
        filename = uploaded_image.name
        # Display the uploaded image
        st.image(uploaded_image, caption='Original Image', use_column_width=True)
        ##Preprocess image
        image, image_flatten = preprocess_image_cnn(Image.open(uploaded_image))
        st.write("Predictions ongoing, please wait...")
        ##Prediction in CNN
        pred_cnn1 = model1.predict(np.expand_dims(image, axis=0))
        result_cnn1 = cnn_model_labels[np.argmax(pred_cnn1)]
        pred_cnn1_aug_train = model1_bal_train.predict(np.expand_dims(image, axis=0))
        result_cnn1_aug_train = cnn_model_labels[np.argmax(pred_cnn1_aug_train)]
        #Prediction in linear models
        predictions_rf = rf_model.predict(image_flatten)
        rf_pred = cnn_model_labels[predictions_rf[0]]
        # Display the predictions
        st.header("Results :")
        st.write("Bellow are results to your skin lesion predictions based on three models. The two first models are CNN models and the third prediction is provided by Random Forest model.")
        st.write("Below are your predictions for your picture", filename)
        st.write(f"CNN Model 1 Prediction: **{result_cnn1}**")
        st.write(f"CNN Model 2 Prediction: **{result_cnn1_aug_train}**")
        st.write(f"Random Forest Prediction: **{rf_pred}**")
if __name__ == "__main__":
    main()
