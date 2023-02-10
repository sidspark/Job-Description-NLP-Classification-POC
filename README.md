## POC for Job Description NLP Classification
---
##### Environment: *Jupyter Notebook* 
##### Language: *Python* | Libraries: *Sklearn, NLTK, SQLite3*
##### Deployment: *Postman*

###### Project Overview: Given a job description, predict which job category it belongs to.
-----
Abbreviation:
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
  
  


