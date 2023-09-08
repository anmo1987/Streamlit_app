##Librairies to import
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
##Import Streamlit
import streamlit as st
import nltk
###FOR NLP FROM NLTK
##from nltk I need
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
##for stopworlds
from nltk.corpus import stopwords
nltk.download('stopwords')
stop_words = stopwords.words("english")
#to transform to token
from nltk.tokenize import word_tokenize
##fromlemmatization & stem
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
##for tagging
from nltk import pos_tag
from nltk.corpus import wordnet
##PorterStemmer
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()

# Librairies for modelling with Gensim
from gensim.models import TfidfModel
from gensim.models import LdaModel
from gensim.models import LsiModel
from gensim.corpora import Dictionary
from gensim.models import LsiModel
from pprint import pprint

###FROM SKLEARN
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise

###for PCA
import matplotlib.pyplot as plt
from sklearn import decomposition
from sklearn.manifold import TSNE

##VISUALISATION
import pyLDAvis
from pyLDAvis import gensim
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
st.set_page_config(
        page_title="NLP analysis from diseases text description",
        page_icon="",  # You can set a custom icon
        layout="wide",  # Use "wide" layout
        initial_sidebar_state="expanded",  # Expand the sidebar by default
        )

#########################INITIALIZE STREAMLIT#######################################################################
def main():
    # Set the page configuration

    st.title("Web application for NLP skin diseases analysis")
    st.write("In this section, we will explore the Disease Text dataSet.")
    st.write("The dataset was obtained from diseases description text from Yale medecine university website. Each section of disease webpage was copyed to a csv file.")
    st.write("**Aim** : Analyse text data using classical NLP tools to discriminate skin diseases.")

    st.write("https://www.yalemedicine.org/")
##Import data
    st.header("Explore the dataframe")
    st.write("**Table 1** : Raw data")
    data = pd.read_csv("/home/annemocoeur/STREAMLIT_APP/NLP_stream/Description_diseases_raw_1.csv", encoding_errors="ignore", delimiter=";")

    st.dataframe(data=data)

    ##PREPROCESSING
    ##Data subset
    var = ["Disease"]
    data_subset =  data[var]
    data_subset["Description_all"] = data["Overview"] + " What : " + data["What"] + " Causes : " + data["Causes"] + " Clinical : " + data["Clinical"] + "Treatment : " + data["Treatment"]

    ###Function to tag word for Lemmenizing
    def get_wordnet_pos(pos_tag):
        output = np.asarray(pos_tag)
        for i in range(len(pos_tag)):
            if pos_tag[i][1].startswith('J'):
                output[i][1] = wordnet.ADJ
            elif pos_tag[i][1].startswith('V'):
                output[i][1] = wordnet.VERB
            elif pos_tag[i][1].startswith('R'):
                output[i][1] = wordnet.ADV
            else:
                output[i][1] = wordnet.NOUN
        return output

    ##Preprocessing NLP
    #Instanciate Lemmatizer
    lemmatizer = WordNetLemmatizer()

    def text_preprocessing(text):
        text = text.lower()
        tokens = word_tokenize(text)
        tokens = [t for t in tokens if t.isalpha()]
        tokens = [t for t in tokens if t not in stop_words]
        pos_tag = nltk.pos_tag(tokens)
        pos_tag = get_wordnet_pos(pos_tag)
        tokens = [lemmatizer.lemmatize(t[0], t[1]) for t in pos_tag]
        tokens = [t for t in tokens if len(t) > 1]
        return tokens

    ##Run Proprocessing function
    data_subset["Description_clean"] = data_subset["Description_all"].apply(lambda x: text_preprocessing(x))
    ##Cleaned Dataset and write to csv cleaned data
    data_NLP  = data_subset.drop("Description_all", axis=1)
    data_NLP = data_NLP.set_index(['Disease'])
    ##Data variable response
    st.header("Text data Preprocessed")
    st.write("Data from all columns are merged within line in a New column.")
    st.write("Text processing is done including:")
    st.write("- keeping alphnumeric words")
    st.write("- removing stop words")
    st.write("- lemmetizing with Tagged words")
    st.write("- tokenizing")


    ##Data SUBSET & PREPROCESSING
    st.write("**Table 2 :** Cleaned dataset")
    st.dataframe(data=data_NLP)

    # TDIDF with SKLEARN
    ###TD-IDF

    vectorizer = TfidfVectorizer(stop_words='english', lowercase=False, analyzer=lambda x: x)
    tf_idf = vectorizer.fit_transform(data_NLP["Description_clean"]).toarray()
    token = vectorizer.get_feature_names_out()
    idf = pd.DataFrame(data = tf_idf, columns=token, index=data_NLP.index)
    ##Data variable response
    st.header("Creating BOW and TD_IDF")
    st.write("From cleaned data, BOW and TD-IDF are created with **Genim** and **Scikit-learn** tools.")
    st.write("TDF-IDF corpus contains **716 words**.")
    st.write("**Table 3:** TF-IDF dataset")
    st.dataframe(data=idf)

    ############SIMILARITIES WITH JACCARD & COSINE

    ##Compute cosine similarities
    cos = pd.DataFrame(pairwise.cosine_similarity(idf), columns=data_NLP.index, index=data_NLP.index)
    #cos.to_csv("Cosine_similarities.csv", header=True, index_label=True)

    ###JACCARD
    def jaquard_similarities_all(Text, data):
        token = list(data_NLP.loc[Text])
        tokens = set(word_tokenize(str(token)))
        ##Struct sources
        Asour = list(data_NLP.loc["Melanocytic nevi"])
        Asourtokens = set(word_tokenize(str(Asour)))
        Bsour = list(data_NLP.loc["Melanoma"])
        Bsourtokens = set(word_tokenize(str(Bsour)))
        Csour = list(data_NLP.loc["Benign keratosis-like lesions"])
        Csourtokens = set(word_tokenize(str(Csour)))
        Dsour = list(data_NLP.loc["basal cell carcinomas"])
        Dsourtokens = set(word_tokenize(str(Dsour)))
        Esour = list(data_NLP.loc["actinic keratosis"])
        Esourtokens = set(word_tokenize(str(Esour)))
        Fsour = list(data_NLP.loc["Dermatofibroma"])
        Fsourtokens = set(word_tokenize(str(Fsour)))
        ##calculate Jaccard
        intercepA = Asourtokens.intersection(tokens)
        unionA = Asourtokens.union(tokens)
        JA = len(intercepA)/len(unionA)
        ##B
        intercepB = Bsourtokens.intersection(tokens)
        unionB = Bsourtokens.union(tokens)
        JB = len(intercepB)/len(unionB)
        ##C
        intercepC = Csourtokens.intersection(tokens)
        unionC = Csourtokens.union(tokens)
        JC = len(intercepC)/len(unionC)
        ##D
        intercepD = Dsourtokens.intersection(tokens)
        unionD = Dsourtokens.union(tokens)
        JD = len(intercepD)/len(unionD)
        ##E
        intercepE = Esourtokens.intersection(tokens)
        unionE = Esourtokens.union(tokens)
        JE = len(intercepE)/len(unionE)
        ##F
        intercepF = Dsourtokens.intersection(tokens)
        unionF = Dsourtokens.union(tokens)
        JF = len(intercepF)/len(unionF)
        res = pd.DataFrame([JA, JB, JC, JD, JE, JF], index = ["Melanocytic nevi", "Melanoma", "Benign keratosis-like lesions", "basal cell carcinomas", "actinic keratosis", "Dermatofibroma"], columns=[Text])
        return res

    ##Calcul Jacquard similarities
    JacML = jaquard_similarities_all("Melanocytic nevi", data_NLP)
    JacM = jaquard_similarities_all("Melanoma", data_NLP)
    JacJKLL = jaquard_similarities_all("Benign keratosis-like lesions", data_NLP)
    JaBCC = jaquard_similarities_all("basal cell carcinomas", data_NLP)
    JaBAK = jaquard_similarities_all("actinic keratosis", data_NLP)
    JaD = jaquard_similarities_all("Dermatofibroma", data_NLP)

    ###Create a dataframe
    simil = pd.concat([JacML, JacM, JacJKLL, JaBCC, JaBAK, JaD], axis=1)


    ##Look for value count
    ##Data variable response
    st.header("Similarities analysis")
    st.write("Cosine similarities (pairwise) and Jaccard distances have been computed from TD-IDF between skin diseases description text.")

    ##PLOTTING JACCARD 

    ###Plotting Heatmapt
    fig, (ax1,ax2, axcb) = plt.subplots(1, 3, figsize=(8, 3), gridspec_kw={'width_ratios':[1,1, 0.08]})
    ax1.get_shared_y_axes().join(ax2)
    g1 = sns.heatmap(cos, annot=True,cbar=False, ax=ax1)
    g1.set_ylabel('')
    g1.set_xlabel('')
    g1.title.set_text('Cosine similarities')
    g2 =sns.heatmap(simil, annot=True, ax=ax2, cbar_ax=axcb)
    g2.set_ylabel('')
    g2.set_xlabel('')
    g2.set_yticks([])
    g2.title.set_text('Jaccard Distances')
    st.pyplot(fig)

    st.write("Similarities are found for most of skin diseases, ranging from 0.1 to 0.34 and 0.11 to 0.23 in Cosine similarities and Jaccard distances, respectively.")
    st.write("However, from these results, Melanocytic nevi and Dermatofibra skin diseases have lower correlations to the others four diseases studies in this dataset.")

    #######PCA
    # TODO: Perform PCA and keep only two dimensions
    ### STRIP_START ###
    ##write idf
    idf.to_csv("TD_IDF_data.csv")
    ##data_pca
    data_pca = pd.read_csv("TD_IDF_data.csv", index_col=0)

    ##Look for value count
    ##Data variable response
    st.header("PCA data analysis")
    st.write("PCA was computed on TF-IDF dataset")


    #####
    pca = decomposition.PCA(n_components=4)
    X_pca = pca.fit_transform(data_pca)
    explained_variance = pca.explained_variance_ratio_


    # Sample data

    # Perform PCA
    pca = decomposition.PCA(n_components=3)
    pca_result = pca.fit_transform(data_pca)
    explained_variance = pca.explained_variance_ratio_

    # Create a DataFrame for Plotly


    # TODO: plot your results!
    ### STRIP_START ###
    fig, ax = plt.subplots()
    ax.scatter(pca_result[:,0],pca_result[:,1])
    for i, txt in enumerate(data_NLP.index):
        plt.annotate(txt, (pca_result[i, 0], pca_result[i, 1]))
    ax.set_xlabel('PC2 (24.5%)')
    ax.set_ylabel('PC2 (23.9%)')
    st.pyplot(fig)
    st.write("PCA plot represents 24.5 and 23.9% of total variance comprises in dataset on PC1 and PC2, respectively.")
    st.write("PCA analysis provides similar results than similarity analysis with the four skin diseases, Benign Kerosis-like lesions, actinic keratosis, being highly correlated also with Basal cell carcinomas and Melanoma. Dermatofibroma and Melanocytic nevi being more independant and less correlated.")

    ########LDA analysis##Look for value count
   #Data variable response
    st.header("Topic analysis")
    st.write("LDA analysis was computed using BOW & TD-IDF matrices and topics results are illustrated using Intertopic distance map.")

    st.header("LDA analysis BOW")
    ##Looking for modelling coherence : Finding the most appropriate number of topics for our corpus
    # Train LDA model.
    # Setting training parameters.
    age = st.slider('Select Number of topics', 2, 10, 2)
    ##Building BOW & TD_IDF with GENSIM
    ##create our corpus
    corpus = data_NLP["Description_clean"]
    ## Compute the dictionary: this is a dictionary mapping words and their corresponding numbers for later visualisation
    id2word = Dictionary(corpus)
    ## Create a BOW
    bow = [id2word.doc2bow(line) for line in corpus]
    ##Compute the TF-IDF
    tfidf_model = TfidfModel(bow)
    # Compute the TF-IDF
    tf_idf_gensim = tfidf_model[bow]
    num_topics = int(age)
    chunksize = 2000
    passes = 20
    iterations = 100
    eval_every = None  

    # Make an index to word dictionary.
    model_bow = LdaModel(
        corpus=bow,
        id2word=id2word,
        chunksize=chunksize,
        alpha='auto',
        eta='auto',
        iterations=iterations,
        num_topics=num_topics,
        passes=passes,
        eval_every=eval_every
    )
    ##Look for the best number of topics
    top_topics_bow = model_bow.top_topics(bow)

    # Average topic coherence is the sum of topic coherences of all topics, divided by the number of topics.
    avg_topic_coherence_bow = sum([t[1] for t in top_topics_bow]) / num_topics
    print('Average topic coherence: %.4f.' % avg_topic_coherence_bow)


    #Visualization of LDA from BOW
    # Visualize the topics
    vis = pyLDAvis.gensim.prepare(topic_model=model_bow, corpus=bow, dictionary=id2word)

    pyLDAvis.save_html(vis, "lda.html")

    from streamlit import components
    with open('./lda.html', 'r') as f:
        html_string = f.read()
    components.v1.html(html_string, width=1300, height=800, scrolling=False)

    st.header("LDA analysis TF-IDF")

    # Train LDA model.
    # Set training parameters.


    num_topics = str(age)
    chunksize = 2000
    passes = 20
    iterations = 100
    eval_every = None  # Don't evaluate model perplexity, takes too much time.

    ##initial model
    #lda1 = LdaModel(corpus=tf_idf_gensim, num_topics=5, id2word=id2word, passes=10, random_state=0)

    # Make an index to word dictionary.
    model_tf = LdaModel(
        corpus=tf_idf_gensim,
        id2word=id2word,
        chunksize=chunksize,
        alpha='auto',
        eta='auto',
        iterations=iterations,
        num_topics=num_topics,
        passes=passes,
        eval_every=eval_every
    )

    vistf = pyLDAvis.gensim.prepare(topic_model=model_tf, corpus=bow, dictionary=id2word)

    pyLDAvis.save_html(vistf, "lda_tf.html")

    from streamlit import components
    with open('./lda_tf.html', 'r') as f:
        html_string = f.read()
    components.v1.html(html_string, width=1300, height=800, scrolling=False)

if __name__ == "__main__":
    main()
    