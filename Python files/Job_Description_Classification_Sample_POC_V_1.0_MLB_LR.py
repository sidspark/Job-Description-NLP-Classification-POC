#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import json
import nltk
import re
import csv
import matplotlib.pyplot as plt 
import seaborn as sns
import tqdm
import pickle

import sklearn

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score

from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, classification_report

import joblib

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))

get_ipython().run_line_magic('matplotlib', 'inline')
pd.set_option('display.max_colwidth', 300)


# In[2]:


def get_top_n_words(corpus: np.ndarray, n: int=10, ngram_range: tuple=(1,3)):
    '''
    Get top common n-grams in corpus.
    
    Parameters:
    corpus: np.ndarray
        Array of texts. n: int (default: 5).ngram_range: tuple (default: (1,3)). Range of n-grams.
    Returns:
        np.ndarray: list of top common n-grams.
        
    '''
    
    tf_idf_vec = TfidfVectorizer(ngram_range=ngram_range, max_features=2000)
    tf_idf_vec.fit(corpus)

    bag_of_words = tf_idf_vec.transform(corpus)

    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in     
                  tf_idf_vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)
    
    return words_freq[:n]


# In[3]:


def load_JD_data():    
    plot_txt_file = "Dataset/IndeedJobsProcessed.xlsx"
    #plot_txt_file = "Dataset/Gov JD Dataset Sample 1.xlsx"
    
    dataframe1 = pd.read_excel(plot_txt_file)
    #dataframe1 = dataframe1[['uniq_id','job_description','category']]
    dataframe1 = dataframe1[['uniq_id','job_description','category_custom']]
    

    dataframe1 = dataframe1[~dataframe1["category_custom"].str.contains('Nothing')]
    dataframe1 = dataframe1[~dataframe1["category_custom"].str.contains('Mechanical')]
    dataframe1 = dataframe1[~dataframe1["category_custom"].str.contains('Consultancy')]
    dataframe1 = dataframe1[~dataframe1["category_custom"].str.contains('Management')]
    dataframe1 = dataframe1[~dataframe1["category_custom"].str.contains('Marketting')]
    
    dataframe1.rename(columns = {'uniq_id':'uniq_id', 'job_description':'job_description',
                              'category_custom':'category'}, inplace = True)

    dataframe1 = dataframe1.groupby('category').filter(lambda x : len(x)>400)
    
    return dataframe1


# In[4]:


def convt_jd_type (jd):
    
    tags = []

    for i in jd['category']:
        tags.append(list(i.split(",")))

    jd['category_new'] = tags
    
    # remove samples with 0 genre tags
    jd_new = jd[~(jd['category_new'].str.len() == 0)]
    
    return jd_new

def data_preprocessing(jd):
    
    jd = convt_jd_type(jd)
    
    return jd
    


# In[5]:


def clean_text(text):
    # remove backslash-apostrophe
    text = re.sub("\'", "", text)
    # remove everything alphabets
    text = re.sub("[^a-zA-Z]"," ",text)
    # remove whitespaces
    text = ' '.join(text.split())
    # convert text to lowercase
    text = text.lower()
    
    #lemmatizer
    wordnet_lemmatizer = nltk.WordNetLemmatizer()
    tokenization = nltk.word_tokenize(text)
    
    text1= []
    for w in tokenization:
        text1.append(wordnet_lemmatizer.lemmatize(w))
    
    text2 = ' '.join(text1)
    
    return text2

# function to remove stopwords
def remove_stopwords(text):
    
    other_words = ['year', 'years', 'work', 'experience', 'job', 'preferred', 'time', 'salary', 'per', 'full',
                   'month', 'day', 'skill', 'skills', 'team', 'shift', 'type', 'apply', 'total', 'good', 'must', 
                  'required', 'internship', 'new']
    
    no_stopword_text = [w for w in text.split() if not w in stop_words]
    text1 = ' '.join(no_stopword_text)
    no_otherwords = [w for w in text1.split() if not w in other_words]
    return ' '.join(no_otherwords)

def cleaning(jd_data):
    
    jd_data['clean_job_description'] = jd_data['job_description'].apply(lambda x: clean_text(str(x)))
    jd_data['clean_job_description'] = jd_data['clean_job_description'].apply(lambda x: remove_stopwords(x))
    
    return jd_data


# In[6]:


def create_features(clean_jd):
    
    multilabel_binarizer = MultiLabelBinarizer()
    multilabel_binarizer.fit(clean_jd['category_new'])

    # transform target variable
    y = multilabel_binarizer.transform(clean_jd['category_new'])
    

    tfidf_vectorizer = TfidfVectorizer() #max_df=0.8, max_features=6)
    
    xtrain, xval, ytrain, yval = train_test_split(clean_jd['clean_job_description'], y, test_size=0.2, random_state=20, stratify=y)
    
    xtrain_tfidf = tfidf_vectorizer.fit_transform(xtrain)
    xval_tfidf = tfidf_vectorizer.transform(xval)
    
    
    return xtrain_tfidf, xval_tfidf, xval, yval, ytrain, multilabel_binarizer, tfidf_vectorizer


# In[7]:


def main():
    #====Load
    jd = load_JD_data()
    
    #====Preprocessing
    jd_data = data_preprocessing(jd)
    #display(jd_data.head())
    
    #====Cleaning
    clean_jd = cleaning(jd_data)
    display(clean_jd.head(5))
    
    #====word frequency
    word_freq_dict = {}
    for query in clean_jd['category'].unique():
        word_freq_dict[query] = get_top_n_words(clean_jd[clean_jd['category']==query]['clean_job_description'], 
                                                n=20, ngram_range=(1,4))
    
    for query in word_freq_dict:
        stat_string = "\n".join([f"{word_freq[0]:35} {word_freq[1]:.2f}" for word_freq in word_freq_dict[query]])
        print(f'''
                ===
            {query}

            {stat_string}
            ''')
    
    #====Feature Engineering
    xtrain_tfidf, xval_tfidf, xval, yval, ytrain, multilabel_binarizer, tfidf_vectorizer = create_features(clean_jd)
    
    #====Train
    lr = LogisticRegression()
    clf = OneVsRestClassifier(lr)
    
    clf.fit(xtrain_tfidf, ytrain)
    
    #----save the model to disk
    filename = 'Save Model Files/JDClassificationPOC_V01.sav'
    joblib.dump(clf, filename)
        
    pickle.dump(tfidf_vectorizer, open("Save Model Files/vectorizerJD_POC_V01.pickle", "wb"))
    pickle.dump(multilabel_binarizer, open("Save Model Files/multilabel_binarizerJD_POC_V01.pickle", "wb"))
    
    #====Prediction
    #----load the model from disk
    tfidf_vectorizer = pickle.load(open("Save Model Files/vectorizerJD_POC_V01.pickle", "rb"))
    multilabel_binarizer = pickle.load(open("Save Model Files/multilabel_binarizerJD_POC_V01.pickle", "rb"))
    loaded_model = joblib.load(filename)
    y_pred = loaded_model.predict(xval_tfidf)
    print_third_prediction = multilabel_binarizer.inverse_transform(y_pred)[3]
    print("Third Prediction: ", print_third_prediction)
    
    #----evaluate performance
    score_f1 = f1_score(yval, y_pred, average="micro")
    print("F1 Score: ",score_f1)
    #----predict probabilities
    y_pred_prob = loaded_model.predict_proba(xval_tfidf)
    t = 0 # threshold value
    y_pred_new = (y_pred_prob >= t).astype(int)
    #----evaluate performance
    f1_score(yval, y_pred_new, average="micro")
    print("F1 Score after threshold of ",t,": ",score_f1)
    
    
    #====Inference and Classification
    def infer_tags(q, tfidf_vectorizer):
        q = clean_text(q)
        q = remove_stopwords(q)
        q_vec = tfidf_vectorizer.transform([q])
        q_pred = loaded_model.predict(q_vec)
        
        lt = multilabel_binarizer.inverse_transform(q_pred)

        out = [item for t in lt for item in t]
        return out
    
    df_test = pd.read_excel("Dataset/Gov JD Dataset Sample Testing 1.xlsx")
    df_test = df_test[['uniq_id','job_description','category']]
    
    for i in range(20):
        k = xval.sample(1).index[0]
        print("uniq_id: ", clean_jd['uniq_id'][k], "\nPredicted Tags: ", infer_tags(xval[k],tfidf_vectorizer)), print("Actual Tags: ",clean_jd['category_new'][k], "\n")   
    
    print("------------TESTING SAMPLE--------------")
    print(df_test['uniq_id'][0])
    print("uniq_id: ", df_test['uniq_id'][0], "\nPredicted Tags: ", infer_tags(df_test['job_description'][0],tfidf_vectorizer)), print("Actual Tags: ",df_test['category'][0], "\n")   
    
# __name__
if __name__=="__main__":
    main()


# In[ ]:





# In[ ]:




