##Import librairies
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os
import random
import seaborn as sns
from PIL import Image
import streamlit as st
###NEURON NETWORK LIBRAIRIES
import tensorflow as tf
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Flatten, Dense, Dropout, BatchNormalization
from keras.utils import to_categorical
from tensorflow.keras import Sequential

##ML LIBRAIRIES
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split



# Set the page configuration
def main():


######################################STREAMLIT ANALYSIS#####################################################################################

    st.title("DATA ANALYSIS AND MACHINE LEARNING ON HAM10000 dataset")


    st.write("On this page, we will provide an overview of the data analysis conducted on the HAM10000 Dataset.")
    st.write("The data analysis will encompass various aspects, including data exploration, preprocessing, visualization, and ML analysis results and insights derived from the dataset.")

##
    st.header("Introduction to HAM10000 Dataset")

    st.write("Dataset Collected from: https://www.kaggle.com/datasets/farjanakabirsamanta/skin-cancer-dataset")


    st.write("**Table 1** : Raw data")
    data = pd.read_csv("DA_ML_CSV_streamlit/DA_dataset/HAM10000_metadata.csv", encoding_errors="ignore")

    st.dataframe(data=data)

    st.write("The column dx countains the different skin diseases studied :")
    st.write("- Actinic keratoses and intraepithelial carcinoma / Bowen's disease (**akiec**)")
    st.write("- basal cell carcinoma (**bcc**)")
    st.write("- benign keratosis-like lesions (solar lentigines / seborrheic keratoses and lichen-planus like keratoses, **bkl**)")
    st.write("- dermatofibroma (**df**)")
    st.write("- melanoma (**mel**)")
    st.write("- melanocytic nevi (**nv**)")
    st.write("- vascular lesions (angiomas, angiokeratomas, pyogenic granulomas and hemorrhage, **vasc**)")
    st.write("dx_type : Histopathology (**histo**) is used to confirm more than 50% of lesions; in the remaining cases, **follow-up** exams, expert **consensus**, or in-vivo **confocal** microscopy confirmation are used as the gold standard (confocal).)")

    st.write("The dataset has been generated synthetically by considering both sex and diseases as factors. The synthesized data are presented in **Table 2**.")
    st.write("**Table 2** : Synthetized")
    data_synthetized = pd.read_csv("DA_ML_CSV_streamlit/DA_dataset/synthetized_data.csv", encoding_errors="ignore")
    st.dataframe(data=data_synthetized)

    st.write("**Data Preprocessing**: In the original dataset, there were missing values within the **age** variable. To address this, missing values were imputed with the mean age value calculated separately for each sex category. Additionally, among the 10,015 observations, 57 were labeled as **Unknown** for the variable **sex**. These  observations were excluded from  dataset to ensure  integrity of the data for subsequent analysis.")

    st.header("Exploratory Data Analysis & Data Visualization")
    st.write("Bellow are some graphical representions of the dataset in order to observe trends across variables.")

#   BOXPLOT
    image = "DA_ML_CSV_streamlit/DA_graphs/Box_plot_skin_age_sex.png"  # Replace with the path to your PNG file
    st.write("**Figure 1 :** Box Plot of diseases type by age and sex.")
    st.image(image, caption="", use_column_width=False)
    st.write("As illustrated in **Figure 1**, there is no significant difference in the distribution of diseases based on sex and age. Notably, melanocytic nevi and vascular lesions exhibit the highest variability and are distributed across a wide range of ages. Conversely, actinic keratoses appear to be more prevalent in individuals aged over 40 years old.")


    # DISEASE REPARTITIONS
    image2 = "DA_ML_CSV_streamlit/DA_graphs/Camembert_diseases_repartition.png"  # Replace with the path to your PNG file

    # Display the image in your Streamlit app
    st.write("**Figure 2 :** Skin diseases distribution in Males and Females.")
    st.image(image2, caption="", use_column_width=False)
    st.write("The distribution of diseases exhibits a strong similarity and correlation between women and men, with the highest frequency of **melanocytic nevi** lesions being observed, and a very low prevalence of **dermatofibroma** and **vascular lesions** within the dataset.")

    ###Diseases localization

    image3 = "DA_ML_CSV_streamlit/DA_graphs/Disease_localization.png"  # Replace with the path to your PNG file

    # Display the image in your Streamlit app
    st.write("**Figure 3 :** Skin diseases lesions localization")
    st.image(image3, caption="", use_column_width=False)

    st.write("A significant majority of the observed lesions are identified as melanocytic nevi, as prominently shown in **Figure 2**. Interestingly, there is a limited presence of lesion types observed on specific body parts such as the neck, hand, foot, acral ear, and scalp.")

    ###Diseases localization
    image4 = "DA_ML_CSV_streamlit/DA_graphs/Camembert_detection_type.png"  # Replace with the path to your PNG file
    st.write("**Figure 4 :** Methods of detection")
    st.image(image4, caption="", use_column_width=False)
    st.write("The majority of the lesions in the dataset have been diagnosed using histopathology (histo) and follow-up approaches, whereas the use of confocal microscopy is infrequent and rarely observed.")


    ###PCA
    image5 = "DA_ML_CSV_streamlit/DA_graphs/PCA_csv.png"  # Replace with the path to your PNG file
    st.write("**Figure 5 :** PCA")
    st.image(image5, caption="", use_column_width=False)
    st.write("Categorical variables have been converted into numerical format to facilitate the computation of Principal Component Analysis (PCA) for dimension reduction. **Figure 5** presents a visual representation of the PCA performed on the first two principal components, which collectively account for 30.32% and 20.59% of the total variance within the dataset, respectively. The observations have been visually represented while taking into account the need to mask the overrepresentation of melanocytic nevi samples.")

    st.header("Preprocess data for Machine Learning classification models")

    st.write("The dataset was modified to prepare it for machine learning (ML) analysis. Categorical variables were processed using the **get_dummies** method to convert them into a numerical format. Additionally, the dataset was split into training and testing subsets using the **train_test_split** function from the sklearn library.")

    st.write("**Table 3:** Dataset used for ML analysis and to generate train and test data.")
    data3 = pd.read_csv("DA_ML_CSV_streamlit/ML_dataset/ML_dataset.csv", index_col=0)
    st.dataframe(data=data3)

    ##Datascaled
    image6 = "DA_ML_CSV_streamlit/ML_graphs_csv/Scaling_data.png"  # Replace with the path to your PNG file
    st.write("**Figure 6 :** Scaling dataset with sklearn standard scaler")
    st.image(image6, caption="", use_column_width=False)

    ##Datascaled
    st.write("Scaled dataset was split in test and train with stratification kept over y_test and y_train.")
    image7 = "DA_ML_CSV_streamlit/ML_graphs_csv/Disease_ratio_train_test.png"  # Replace with the path to your PNG file
    st.write("**Figure 7 :** Skin diseases distribution in train and test dataset")
    st.image(image7, caption="", use_column_width=False)

    st.header("Machine Learning classification models")

    st.write("In total, eight machine learning classification models were evaluated using a dedicated function. The classification results for each of these models on both the training and testing populations are presented in **Table 4**.")

    st.write("ML Models tested:")
    st.write("- Logistic regression")
    st.write("- Descision tree classifier")
    st.write("- Random forest classifier")
    st.write("- Gradiant boosting classifier")
    st.write("- AdaBoost")
    st.write("- Linear discriminant analysis")
    st.write("- SVC")
    st.write("- Ridge classifier")

    ##Table results ML
    st.header("Results and Insights")

    st.write("**Table 4:** Results obtained from the 8 ML classification models.")
    data4 = pd.read_csv("DA_ML_CSV_streamlit/ML_dataset/Results_ML_csv.csv", index_col=0)
    st.dataframe(data=data4)

    ##Results ML
    image8 = "DA_ML_CSV_streamlit/ML_graphs_csv/Results_ML_models_csv.png"  # Replace with the path to your PNG file
    st.write("**Figure 8 :** Plotting accuracy for the 8 models tested")
    st.image(image8, caption="", use_column_width=False)

    st.write("From **Figure 8** and  **Table 4** we can see that best accuracy scores were obtained from **Descision Tree** and **Random Forest classifier**.")

    ##Classification report
    image9 = "DA_ML_CSV_streamlit/ML_graphs_csv/Classification_reports_results_ML_csv.png"  # Replace with the path to your PNG file
    st.write("**Figure 9 :** Classification report results for the 8 ML classification models.")
    st.image(image9, caption="", use_column_width=False)

    ##Results on diseases
    image10 = "DA_ML_CSV_streamlit/ML_graphs_csv/results_precision_ML_csv.png"  # Replace with the path to your PNG file
    st.write("**Figure 10 :** Classification report results for the 8 ML classification models.")
    st.image(image10, caption="", use_column_width=False)


    ##Results on diseases
    image12 = "DA_ML_CSV_streamlit/ML_graphs_csv/ConfusionMetrices_ML_csv.png"  # Replace with the path to your PNG file
    st.write("**Figure 11 :** Confusion matrices on the three best ML models.")
    st.image(image12, caption="", use_column_width=False)
    st.write("All three models demonstrate excellent performance in detecting the three main represented skin diseases. However, for the less common diseases, it is evident that the models tend to produce a significant number of false negatives. This observation is likely due to the **high class imbalance** present in the dataset. To mitigate this issue and enhance accuracy, oversampling techniques such as **SMOTE** (Synthetic Minority Over-sampling Technique) could be applied to balance the class distribution and improve the models' ability to detect the less common diseases.")

    st.header("Optimizing results with GridCV")
    st.write("Hyperparameter optimization was carried out for both the Decision Tree and Random Forest classifiers. However, as indicated by the results presented in **Table 5**, the highest accuracies were achieved using the default hyperparameters, suggesting that the default settings provided the best performance for these models.")

    data5 = pd.read_csv("DA_ML_CSV_streamlit/ML_dataset/Results_grid.csv", index_col=0)
    st.dataframe(data=data5)


    st.write("**The random Forest weights were saved for the final detection app.**")

if __name__ == "__main__":
    main()
