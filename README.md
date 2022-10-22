# Pneumonia Detection Using Transfer Learning

* Pneumonia is an infectious and fatal respiratory disease caused by bacteria, fungi, or a virus that infects human lung air sacs with a load full of fluid or pus.

* Chest x-rays are the common method used to diagnose pneumonia and it takes a medical expert to assess the result of the x-ray. The troublesome method of detecting pneumonia leads to loss of life due to improper diagnosis and treatment.

* With the emerging computing power, the development of an automatic pneumonia detection and disease treatment system is now possible, especially if the patient is in a remote area and medical services are limited.

* In this Web application, we used Transfer Learning model [**VGG16**](https://keras.io/api/applications/vgg/) for prediction

* This Web application is created and deployed in [**Streamlit**](https://streamlit.io/)

## 💡 How To Use Our Web Application

![alt text](https://raw.githubusercontent.com/mvram123/mvram123/main/Pneumonia/Webapp.png)

**Step 1**: Click this link to visit our web application: [Pneumonia Detection](https://share.streamlit.io/mvram123/pneumonia-detection/main/app.py)

**Step 2**: Enter the Image URL which you want to clasify and click `Enter`

**Step 3**: That's it !! You will get the output.



## ⏳ Data

![Illustrative Examples of Chest X-Rays in Patients with Pneumoni](https://i.imgur.com/jZqpV51.png)

The normal chest X-ray (left panel) depicts clear lungs without any areas of abnormal opacification in the image. Bacterial pneumonia (middle) typically exhibits a focal lobar consolidation, in this case in the right upper lobe (white arrows), whereas viral pneumonia (right) manifests with a more diffuse ‘‘interstitial’’ pattern in both lungs.

The dataset I’m using here is stored as .jpg files in 2 different folders one is named with Normal which consists of normal chest x-ray Images and other is named with Pneumonia which consists of pneumonia images. 

Chest X-ray images (anterior-posterior) were selected from retrospective cohorts of pediatric patients of **one to five years old** from **Guangzhou Women and Children’s Medical Center, Guangzhou**. All chest X-ray imaging was performed as part of patients’ routine clinical care.

For the analysis of chest x-ray images, all chest radiographs were initially screened for quality control by removing all low quality or unreadable scans. The diagnoses for the images were then graded by two expert physicians before being cleared for training the AI system. In order to account for any grading errors, the evaluation set was also checked by a third expert.

Dataset Link (Taken from Kaggle): https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia

## 📝 Project Architecture

### Machine Learning Pipelines That need to constructed

1. Data Collection - Directly From Kaggle
2. Data Validation - Optional
3. Data Preprocessing / Feature Engineering - Done
4. Model Training - Done
5. Model Evaluation - Done
6. Web Application Creation - Done
7. Testing - **Not Done**


### Automatic Scripts that need to be constructed

1. CI-CD Deployment - Not Done
2. Model Monitoring - Not Done
3. Model Retraining Scripts - Not Done

### Model Artifacts which need to be stored

#### For Machine Learning Model (All Experiments)

1. Model Parameters - Done
2. Model Summary - Done
3. Model Performance metrics - Done
4. Model Location and libraries we used - Done

#### For Data

1. Data Schema - Not neccessary
2. Data Collection Locations (Can be Multiple) - Kaggle Website
3. Data Storage Location - Local System
4. Data Features, Feature Distributions, Feature labels etc - Optional

## 🖥️ Libraries Used

* Tensorflow
* Keras
* Scikit-learn
* Streamlit

## 🧑🏼‍💻 Contributors

1. M V Ramarao


