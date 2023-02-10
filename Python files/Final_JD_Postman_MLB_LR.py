#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import json
import nltk
import re
import csv
import joblib
import pickle
import flask
from flask import Flask, request, jsonify
import os

import spacy
import sqlite3

from nltk.stem import WordNetLemmatizer

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))


# In[2]:


#Initiating Flask -----------------------------------------------------------------------
app = Flask(__name__)


# In[3]:


#Loading saved ML files ------------------------------------------------------------------
tfidf_vectorizer = pickle.load(open("Save Model Files/vectorizerJD_POC_V01.pickle", "rb"))
multilabel_binarizer = pickle.load(open("Save Model Files/multilabel_binarizerJD_POC_V01.pickle", "rb"))
loaded_model = joblib.load('Save Model Files/JDClassificationPOC_V01.sav')


# In[4]:


#Predicting -------------------------------------------------------------------------------
def infer_tags(q, tfidf_vectorizer):
    q = clean_text(q)
    q = remove_stopwords(q)
    q_vec = tfidf_vectorizer.transform([q])
    q_pred = loaded_model.predict(q_vec)

    lt = multilabel_binarizer.inverse_transform(q_pred)

    out = [item for t in lt for item in t]

    return out

#Clean text -------------------------------------------------------------------------------
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

#Function to remove stopwords ----------------------------------------------------------------
def remove_stopwords(text):
    
    no_stopword_text = [w for w in text.split() if not w in stop_words]
    
    return ' '.join(no_stopword_text)

#Inference and Classification -----------------------------------------------------------------
def infer_tags(q, tfidf_vectorizer):
    
    q = clean_text(q)
    q = remove_stopwords(q)
    q_vec = tfidf_vectorizer.transform([q])
    q_pred = loaded_model.predict(q_vec)

    lt = multilabel_binarizer.inverse_transform(q_pred)

    out = [item for t in lt for item in t]
    
    return out

#NLP Tokens ------------------------------------------------------------------------------------
def ner_text(text):

    nlp = spacy.load('en_core_web_sm')
    print(nlp._path)
    
    ner_tokens = []
    doc = nlp(text)
  
    for ent in doc.ents:
        ner_tokens.append(ent.text+"->"+ent.label_)

    return ner_tokens


# In[5]:


#Insert into Applicant DB ---------------------------------------------------------------------
def insert_applicant_db(logindf_from_post):
    
    sqliteConnection = sqlite3.connect('Database\job_applicants.db')
    cursor = sqliteConnection.cursor()
    
    try:
        df = pd.DataFrame([['#',str(logindf_from_post['name'][0]),
                            logindf_from_post['number'][0],
                          str(logindf_from_post['job_description'][0]),
                          str(logindf_from_post['address'][0])]],
                          columns=['uniq_id','name','number','job_description','address'])
        
        df.to_sql('job_applicants', sqliteConnection, if_exists='append', index = False)
    
    except:
        return ''
    
    #Close DB        
    if sqliteConnection:
        sqliteConnection.close()
        print('SQLite Connection to applicant DB closed')
    
    return

#Insert into Login DB --------------------------------------------------------------------------
def insert_login_db(logindf_from_post):
    
    sqliteConnection = sqlite3.connect('Database\job_applicants_login.db')
    cursor = sqliteConnection.cursor()
    
    #If name not in db
    try:
        df = pd.DataFrame([['#',str(logindf_from_post['name'][0]),
                            str(logindf_from_post['password'][0])]], 
                          columns=['uniq_id','name','password'])
        df.to_sql('job_applicants_login', sqliteConnection, if_exists='append', index = False)
    
    except:
        return ''
    
    #Close DB        
    if sqliteConnection:
        sqliteConnection.close()
        print('SQLite Connection to login DB closed')
    
    return

#Login DB --------------------------------------------------------------------------------------
def login_db(post_name):
    
    sqliteConnection = sqlite3.connect('Database\job_applicants_login.db')
    cursor = sqliteConnection.cursor()
    
    #If name not in db
    try:
        select_all = "SELECT * FROM job_applicants_login WHERE name = '"+post_name+"'"
        info_from_db = cursor.execute(select_all).fetchall()
        pass_df = pd.DataFrame(info_from_db, columns = ['uniq_id','name','password'])
        pass_from_df = pass_df['password'][0]
    
    except:
        return ''
    
    #Close DB        
    if sqliteConnection:
        sqliteConnection.close()
        print('SQLite Connection to login DB closed')
        
    return pass_from_df

#Applicant DB ----------------------------------------------------------------------------------
def job_applicants_db(login_name_from_post):

    sqliteConnection = sqlite3.connect('Database\job_applicants.db')
    cursor = sqliteConnection.cursor()
    
    select_all = "SELECT * FROM job_applicants WHERE name = '"+login_name_from_post+"'"
    
    rows = cursor.execute(select_all).fetchall()
    
    querydf = pd.DataFrame(rows, columns = ['uniq_id','name','number', 'job_description','address'])
    
    #Close DB
    if sqliteConnection:
        sqliteConnection.close()
        print('SQLite Connection to applicants JD closed')
        
    return querydf

#Job Description DB -------------------------------------------------------------------------------
def job_description_db(prediction, inter_json):
    
    # Connect to DB and create a cursor
    sqliteConnection = sqlite3.connect('Database\job_description.db')
    cursor = sqliteConnection.cursor()

    # Fetch and output result
    query = "SELECT * FROM job_description WHERE category = '"+prediction[0]+"' LIMIT 2"

    rows = cursor.execute(str(query)).fetchall()

    for r in rows:
        inter_json.append({"Unique_JD_ID": r[0],
                "Job_Description":r[1], 
                "Category": r[2]})

    # Committing the changes
    sqliteConnection.commit()

    #Close DB
    if sqliteConnection:
        sqliteConnection.close()
        print('SQLite Connection to JD DB closed')
    
    return inter_json


# In[6]:


@app.route('/predict', methods=['POST'])

def predict():
    
    #login DB ----------------------------------------------------------------------------------------------
    sqliteConnection = sqlite3.connect('Database\job_applicants_login.db')
    cursor = sqliteConnection.cursor()
    
    json_ = request.json
    logindf_from_post = pd.DataFrame(json_)
    
    #Get data from login db --------------------------------------------------------------------------------
    try:
        pass_from_df = login_db(logindf_from_post['name'][0])
        if pass_from_df == '':
            if logindf_from_post['register'][0] == 'yes':
                #insert new user login
                insert_login_db(logindf_from_post)
                insert_applicant_db(logindf_from_post)
                return jsonify([{"Message":"Created new credentials. Please run the same POST again"}])
            return jsonify([{"Message":"Unable to authenticate. User name does not exist. Please create new credentials."}])
    #Handle Error
    except sqlite3.Error as error:
        print('Error occured connecting to login_db - ', error)
    
    login_name_from_post = logindf_from_post['name'][0]
    passwrd_from_post = logindf_from_post['password'][0]
    
    #Password Authenticate ----------------------------------------------------------------------------------
    if str(passwrd_from_post) == str(pass_from_df):
        print("Authenticated")
    else:
        print("Pass Incorrect")
        return jsonify([{"Message":"Unable to authenticate. Please enter valid password."}])
    
    #Get data from applicants db -----------------------------------------------------------------------------
    try:
        querydf = job_applicants_db(login_name_from_post)
    #Handle Error
    except sqlite3.Error as error:
        print('Error occured connecting to job_applicants_db - ', error)
    
    final_json = []
    inter_json = []
    
    #Prediction ----------------------------------------------------------------------------------------------
    for i, row in querydf.iterrows():
        text = querydf['job_description'][i]
        ids = querydf['uniq_id'][i]
        name = querydf['name'][i]
        number = querydf['number'][i]
        address = querydf['address'][i]

        ner_tokens = ner_text(text)

        prediction = infer_tags(str(text),tfidf_vectorizer)
        response = list(prediction)
        
        #If no prediction
        if not response:
            prediction = 'Unable categorize JD.'
            response = [prediction]
        
        #Connect to job description db ----------------------------------------------------------------------
        try:
            inter_json = job_description_db(prediction, inter_json)
        #Handle Error
        except sqlite3.Error as error:
            print('Error occured connecting to job_description_db - ', error)

            
        # Close DB Connection irrespective of success or failure ---------------------------------------------
        finally:

            if sqliteConnection:
                sqliteConnection.close()

            final_json.append({"Unique_Applicant_ID": str(ids),
                                "Name": name,
                                "Phone Number": str(number),
                                "Address": address,
                                "Applicant_Job_Description":text, 
                                "Prediction": response,
                                "NER_Tokens": ner_tokens,
                                "Similar_Jobs": inter_json})
            
            inter_json = []

    return jsonify(final_json)

if __name__ == '__main__':
    
    app.run(debug=False)


# , 
#     {     
#     "unique_id": "ID 02", 
#     "job_description": "Planning concepts by studying relevant information and materials. Illustrating concepts by designing examples of art arrangement, size, type size and style and submitting them for approval.Preparing finished art by operating necessary equipment.Coordinating with outside agencies, art services, web designer, marketing, printers, and colleagues as necessary.Contributing to team efforts by accomplishing tasks as needed.Communicating about layout and design.Creating a wide range of graphics and layouts for product illustrations, company logos, and using photoshop.Reviewing final layouts and suggesting improvements when necessary.Graphic Designer Requirements: Degree in graphic design.Experience as a graphic designer or in related field.Demonstrable graphic design skills with a strong portfolio.Proficiency with Photoshop, InDesign Quark, and Illustrator.A strong eye for visual composition.Effective time management skills and the ability to meet deadlines.Able to give and receive constructive criticism." 
#     }, 
#     { 
#     "unique_id": "ID 03", 
#     "job_description": "Understand how to identify new sales opportunities and follow through. Work closely with customers and maintain good relations. Work in a fast paced goal oriented environment. Learn on the job regarding how to conduct demos and sales in general. With time this job has a scope for growth. Responsibilities Demo and sell products to existing and prospective customers Establish and maintain positive business and customer relationships. Achieve sales targets and outcomes. Collaborate with team members and other departments Generate reports on new sales, target achievements, issues to the management"
#     }, 
#     { 
#     "unique_id": "ID 04", 
#     "job_description": "Understand how to"
#     }    

# In[ ]:




