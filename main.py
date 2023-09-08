import streamlit as st
from StreamlitImageDetection import app_detection
from DA_ML_CSV_streamlit import app_da
from NLP_stream import app_nlp
from CNN_streamlit import app_cnn


def main():
    st.title("")

    # Create a sidebar navigation menu
    selected_page = st.sidebar.selectbox("Select a Page", ["Image Detection App", "HAM10000 dataset analysis DA/ML", "CNN model analysis", "NLP analysis from diseases text description"])

    # Display the selected page content
    if selected_page == "Image Detection App":
        app_detection.main()
    elif selected_page == "HAM10000 dataset analysis DA/ML":
        app_da.main()
    elif selected_page == "CNN model analysis":
        app_cnn.main()
    elif selected_page == "NLP analysis from diseases text description":
        app_nlp.main()

if __name__ == "__main__":
    main()
