## POC for Job Description NLP Classification
---
##### **Environment**: *Jupyter Notebook* 
##### **Language**: *Python* | Libraries: *Sklearn, NLTK, SQLite3, Flask*
##### **Deployment**: *Postman*

###### **Project Overview**: Given a job description, predict which job category it belongs to.
-----
### Before We Start

Abbreviations:
- Techinal:
  - MLB : Multi Label Binarizer
  - LR : Logistic Regression
- Non-Technical:
  - JD : Job Description
  - POC : Proof of Concept

File Descriptions:
- Main Directory:
  - Using MLB and LR
    - Train and test file: MLB_LR-JD_Classification_POC.ipynb
    - Postman Deployment file: MLB_LR-Postman_JD.ipynb
- Dataset Folder:
  - Train data: IndeedJobsProcessed.xlsx (480 samples from 6 different categories)
  - Test data: JDTesting.xlsx (1 record for testing)
- Database Folder:
  - Dummy Login Info: job_applicants_login.db (2 users)
  - Applicants Info: job_applicants.db (w.r.t 2 user from above)
  - Extra Job Descriptions: job_description.db (25 records per category)
- Save Model Files Folder:
  - Saving MLB LR
    - Trained Model: JDClassificationPOC_V01.sav
    - Saved Weights for MLB: multilable_binarizerJD_POC_V01.pickle
    - Saved Weights for TF-IDF: vectorizerJD_POC_V01.pickle
----
  
### Process
##### Training

- Simple run the train and test file: MLB_LR-JD_Classification_POC.ipynb in your environment.
- It takes train data: IndeedJobsProcessed.xlsx → preprocesses → creates features → splits (0.2) → trains → saves model and weights.
- The preprocessing steps takes care of:
  - Cleaning: takes are of whitespaces, converts to lowercase, keep alphabets only, etc.
  - Stopwords: NLTK library.
  - 'Other' words: words that are common. For example: job, year, etc.
  - Lemmatization and Tokenization.
- Frquency of each words w.r.t to its categories is displayed for better understanding.

##### Testing

- The trained data is then validated again the remaining data (0.2).
- Model predicts the test data: JDTesting.xlsx.
- Model also predicts and displays 20 random sample from the validation data itself.

#### F1 Score: ***72.84***
---
##### Deploying it in Postman
##### ***Method: POST*** | ***URL: localhost:5000/predict***

Sample json input for Postman:
```
[
  {
    "name": "John Doe",
    "password": "JohnDoes@123#",
    "register": "No",
    "_comment": "Below only activates is the user is new and the register is set to yes",
    "number": 8177774344,
    "job_description": "duties such as implementing behavior guidelines, preparing class activities, using various teaching methods, assigning homework, giving tests, and monitoring student academic performance. They may also be required to manage classroom materials and inventory.",
    "address": "2 Abes, 12 Lincon Rd., Fakesville."
  }
]
```



