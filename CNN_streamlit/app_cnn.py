import os
import pandas as pd
from PIL import Image
import numpy as np
import random
import seaborn as sns
##from sklearn
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder
from matplotlib import pyplot as plt
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical

### sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import classification_report, accuracy_score
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.linear_model import RidgeClassifier

### tensorflow
from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras import layers, models
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import MaxPooling2D, Conv2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import streamlit as st
###RUN FUNCTION
def main():
    
    st.title("Image Analysis CNN models")
    st.write("On this page, we will provide an overview of analysis conducted on the HAM10000 pictures.")
    st.write("A training set for academic machine learning can be created using the dataset, which comprises of 10015 dermatoscopic images. All significant diagnostic categories for pigmented lesions are represented in the cases in a representative manner:")
    st.write("The column dx countains the different skin diseases studied :")
    st.write("- Actinic keratoses and intraepithelial carcinoma / Bowen's disease (**akiec**)")
    st.write("- basal cell carcinoma (**bcc**)")
    st.write("- benign keratosis-like lesions (solar lentigines / seborrheic keratoses and lichen-planus like keratoses, **bkl**)")
    st.write("- dermatofibroma (**df**)")
    st.write("- melanoma (**mel**)")
    st.write("- melanocytic nevi (**nv**)")
    st.write("- vascular lesions (angiomas, angiokeratomas, pyogenic granulomas and hemorrhage, **vasc**)")
    st.write("dx_type : Histopathology (**histo**) is used to confirm more than 50% of lesions; in the remaining cases, **follow-up** exams, expert **consensus**, or in-vivo **confocal** microscopy confirmation are used as the gold standard (confocal).)")
    
    st.header("EDA and propressed data")
    st.write("The 10015 pictures are in format (450, 600, 3).")
    
    ##Import images
    image = "CNN_streamlit/CNN_graphs/Photo_maladie.png"
    st.write("**Figure 1 :** some random pictures of the seven skin lesions.")
    st.image(image, caption="", use_column_width=False)
    st.write("Pictures were imported in format (28,28,3) for CNN modelling, then scaled. Scaled pictures were also flattened for ML models. The y variable was found on the dx variable from HAM10000 dataset. The target variable was converted to categorical for CNN analysis and classes ranged from 0 to 6 for ['akiec' 'bcc' 'bkl' 'df' 'mel' 'nv' 'vasc']. Train and test dataset was generated with 20% of the original dataset for the test and with stratification kept.")

    ##CNN MODELS
    st.header("CNN Models")
    st.write("Three different CNN models were used. Bellow are model descriptions.")
    st.write("**Model  1 :  Sequential**")
    st.write("- Conv2D layer with 32 kernels of size (3,3) & input shape 28,28,3, activation function ReLu")
    st.write("- Maxpooling 2D 2,2")
    st.write("- Conv2D with 64 kernels sized (3,3), activation function ReLu")   
    st.write("- Maxpooling 2D 2,2")
    st.write("- Conv2D with 64 kernels sized (3,3), activation function ReLu")
    st.write("- Flatten images")
    st.write("- Dense layers 64 units, ReLu activation function")
    st.write("- Dense layer 7 units, ReLu activation function")
    st.write("- Output Dense Layers, with 7 units for the 7 skin cancers categories and Softmax activation")
    
    st.write("**Model LeNet5 :  Sequential**")
    st.write("- Conv2D layer with 6 kernels of size (5,5) & input shape 28,28,3, activation function ReLu")
    st.write("- Maxpooling 2D 2,2")
    st.write("- Conv2D with 16 kernels sized (5,5), activation function ReLu")   
    st.write("- Maxpooling 2D 2,2")
    st.write("- Flatten images")
    st.write("- Dense layers 120 units, ReLu activation function")
    st.write("- Dense layer 84 units, ReLu activation function")
    st.write("- Output Dense Layers, with 7 units for the 7 skin cancers categories and Softmax activation")

    st.write("**Model 3 :  Sequential**")
    st.write("- Conv2D layer with 6 kernels of size 5 & input shape 28,28,3, activation function ReLu")
    st.write("- Maxpooling 2D 2,2")
    st.write("- Conv2D with 16 kernels sized 5, activation function ReLu")   
    st.write("- Maxpooling 2D 2,2")
    st.write("- Flatten images")
    st.write("- Dense layers 120 units, ReLu activation function")
    st.write("- Dense layer 84 units, ReLu activation function")
    st.write("- Output Dense Layers, with 7 units for the 7 skin cancers categories and Softmax activation")
    st.write("- kernel_regularization l2 added for each Conv2D and Dense layer.")
    
    st.write("**Model compilation** : each model was compiled with adam optimizer, categorical crossentropy loss and accuracy was measured.")
    st.write("**Model fitting** : each model was fitted using 25 epochs, 202 batch and a call back at patience = 10.")
    ##Results CNN
    st.header("Results from CNN")
    image2 = "/CNN_streamlit/CNN_graphs/Results_CNN_models.png"
    st.write("**Figure 2 :** Results obtained on the accuracy in test and train dataset over the 25 epochs for the three CNN model.")
    st.image(image2, caption="", use_column_width=False)
    st.write("According to results illustrated in **Figure 2**, best results were obtained with **Model 1**, however after the 18 epochs the model started to overfit on train data.")
    
    ##Data augmentation
    st.header("Dataset augmentation")
    st.write("Original dataset was modifyed and pictures were **augmented** using **ImageDataGenerator** from tensorflow. Parameters passed through the Image generator included:")
    st.code("""
    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        brightness_range=(0.5, 1.5))""")
    st.write("Train dataset was fited with Image generator at batch_size=32 and the three models were computed and fited again with the new train dataset augmented.")
    ##results data augmentation
    image3 = "/CNN_streamlit/CNN_graphs/CNN_Results_Data_Augmentation.jpg" 
    st.write("**Figure 3 :** Results obtained on the accuracy in test and train dataset over the 25 epochs for the three CNN model.")
    st.image(image3, caption="", use_column_width=False)
    st.write("Really bad results were obtained as seen in **Figure 3** appart from Model 3 which was highly overfitting from very first epoch. Therefore it seems that augmentation on our model **did not** improved accuracy and performance.")
    ##Balancing data
    st.header("Balancing Dataset")
    st.write("As already notified from analysis of the HAM10000.csv file in the previous section, categorical dataset was highly imbalanced with over representation of one classe. For improving model performance, oversampling was done using **SMOTE** on train and test data.")
    st.code("""
    from imblearn.over_sampling import SMOTE
    smote = SMOTE(sampling_strategy='auto', random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    Class distribution after SMOTE resampling: {0: 6705, 1: 6705, 2: 6705, 3: 6705, 4: 6705, 5: 6705, 6: 6705}""")
    ##Results balanced dataset
    image4 = "/CNN_streamlit/CNN_graphs/CNN_Results_Balanced_data.jpg" 
    st.write("**Figure 4 :** Results obtained on the accuracy in oversampled dataset over  25 epochs for two CNN models selected (Model1 & Model3).")
    st.image(image4, caption="", use_column_width=False)
    st.write("Very good results were obtained on both models with data oversampled. However test dataset was also oversampled, explaining very good accuracy on the test dataset. CNN Model 1 was reran with oversampled data only for the train.")
    st.code("""Epoch 25/25 : 40s 215ms/step - loss: 0.1603 - accuracy: 0.9457 - val_loss: 0.4664 - val_accuracy: 0.8313""")
    st.write("The two model weight, original **Model 1** and **Model 1 oversampled on train** were saved for latter predictions.")

    ##ML
    st.header("Machine Learning models")
    st.write("Classical ML classification models were also tested on the flattened pictures dataset, with split train and test of 80 / 20%.")
    st.write("**ML models tested** : Descision Tree, Random Forest, K Neighbors classifier, Linear Discriminant Analysis, SVM")
    ##Results ML
    image5 = "/CNN_streamlit/CNN_graphs/ML_results_imagesanalysis.png" 
    st.write("**Figure 5 :** Results obtained with ML classification models on flattened pictured.")
    st.image(image5, caption="", use_column_width=False)
    st.write("Best results from ML were obtained with **Random Forest model**. Accuracy ranged from 0.62 to 0.72 in ML compared to 0.7 to 0.8 in CNN Models.")
    st.write("Confusion matrices were computed from results obtained in test with the Logistic regression and Random forest model as shown in **Figure 6**.")
    ##Confusion matrices from ML
    image6 = "/CNN_streamlit/CNN_graphs/confusion_matrice_ML_images_analysis.png" 
    st.write("**Figure 6 :** Confusion matrices obtained in test data.")
    st.image(image6, caption="", use_column_width=False)
    st.write("Based on **Figure 6**, results in test are more accurate in the Random Forest model.")
    ##Conclusion
    st.header("Conclusion")
    st.write("For the detection app, predictions will be computed using **Model 1**, **Model 1 with oversampling on train** and **ranfom forest** .")
if __name__ == "__main__":
    main()
