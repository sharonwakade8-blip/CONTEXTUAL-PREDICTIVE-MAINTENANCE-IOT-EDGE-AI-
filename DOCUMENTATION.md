### Contextual Predictive Maintenance (IoT Edge AI) 

## Team Members and Responsibilities :

## 1. Sharon Wakade – Project Leader & ML Integration 
### Responsibilities :
* Plan the overall project.
* Assign tasks to team members.
* Coordinate team progress.
* Integrate all project modules.
* Manage the GitHub repository.
* Train the final LightGBM model.
* Evaluate the model performance.
* Prepare the README file.
* Prepare the project documentation.
* Write the final project report.

### Key Deliverables :
* main.py
* README.md
* Final documentation
* Final report (PDF)
* Integrated project
* GitHub repository
------

## 2. Sakshi jamadar – Data Collection & Preprocessing Engineer 
### Responsibilities :
* Collect IoT sensor datasets.
* Collect weather datasets.
* Handle missing values.
* Remove duplicate records.
* Detect and handle outliers.
* Format timestamps.
* Normalize data.
* Build the preprocessing pipeline.

### Key Deliverables:
* data_ingestion.py
*  preprocessing.py
* Cleaned dataset
* Processed dataset
------

## 3. Samruddhi Naikode – Feature Engineering Engineer 
### Responsibilities : 
* Perform Exploratory Data Analysis (EDA).
* Create rolling statistical features.
* Generate lag features.
* Create trend features.
* Select important features.
* Prepare the final feature dataset.

### Key Deliverables: 
* feature_engineering.py
* Feature dataset
* EDA notebooks
-----

## 4.  Avinash k  – Data Fusion & Context Integration Engineer 
### Responsibilities :  
* Collect external weather data.
* Merge weather and sensor datasets.
* Synchronize timestamps.
* Handle missing contextual values.
* Build the contextual data fusion pipeline. 

### Key Deliverables:  
* data_fusion.py
* merged_data.csv
-----

## 5.  Venkatesh cs  – Machine Learning Engineer 
### Responsibilities :  
* Apply SMOTE for class balancing.
* Split training and testing datasets.
* Train the LightGBM model.
* Tune model hyperparameters.
* Perform cross-validation.
* Evaluate model performance.
* Save the trained model and scaler.

### Key Deliverables:  
* train.py
* evaluate.py
* lightgbm_model.pkl
* scaler.pkl
* Evaluation metrics and reports
------

 ## 6.  vishnu ES  – Dashboard & Deployment Engineer 
 ### Responsibilities :   
* Build the Streamlit dashboard.
* Display prediction results.
* Show machine health status.
* Create alerts and notifications.
* Visualize feature importance.
* Test the application.
* Support deployment.

### Key Deliverables: 
* dashboard/app.py
* predict.py
* Dashboard assets
* Testing files
-----


## Final Integrated Workflow
* Data Collection
* Data Preprocessing
* Feature Engineering
* Contextual Data Fusion
* Model Training (LightGBM + SMOTE)
* Model Evaluation
* Prediction Generation
* Streamlit Dashboard
* Project Integration
* Documentation, GitHub & Final Presentation

Data Collection
      ↓
Data Preprocessing
      ↓
Feature Engineering
      ↓
Contextual Data Fusion
      ↓
Model Training
      ↓
Model Evaluation
      ↓
Prediction
      ↓
Streamlit Dashboard
      ↓
Documentation & Final Submission

-------


## Abstract :
Predictive maintenance uses machine learning to identify potential equipment failures before they occur. This project combines IoT sensor data with contextual weather information to predict machine failures using a LightGBM model. The system includes data preprocessing, feature engineering, model training, evaluation, and a Streamlit dashboard for real-time prediction visualization. 

## Introduction :
Industrial equipment generates large amounts of sensor data. Traditional maintenance approaches either waste resources through unnecessary servicing or risk failures by reacting too late. Predictive maintenance leverages AI and machine learning to analyze sensor data and predict failures in advance, helping reduce downtime and maintenance costs. 

## Problem Statement :
Develop a machine learning system that predicts machine failures using IoT sensor data and environmental conditions, allowing maintenance teams to take preventive action before equipment breaks down. 

## Technologies Used:
Python
Pandas
NumPy
Scikit-learn
LightGBM
Streamlit
Plotly
Matplotlib
GitHub 

## Objectives:
Collect and preprocess IoT sensor data.
Handle missing values and normalize data.
Perform feature engineering.
Merge sensor and weather datasets.
Train a LightGBM classification model.
Evaluate model performance.
Develop a Streamlit dashboard for predictions. 

## Dataset Description:
The project uses:
ai4i_predictive_maintenance.csv – Main IoT sensor dataset.
sensor_data.csv – Processed sensor readings.
weather_data.csv – Contextual weather information.
merged_data.csv – Final merged dataset used for training.

## System Architecture:
Data Collection
Data Preprocessing
Feature Engineering
Data Fusion
Model Training
Model Evaluation
Prediction
Streamlit Dashboard .

##  Methodology:
## Data Preprocessing
Remove missing values
Remove duplicate records
Sort by timestamp
Standardize features

## Feature Engineering
Rolling mean of vibration
Rolling standard deviation
Lag features
Temperature lag

## Data Fusion
Merge sensor data with weather information
Align timestamps

## Model Training
Split training and testing datasets
Train a LightGBM classifier
Save trained model

## Model Evaluation
Accuracy
Precision
Recall
F1-Score
ROC-AUC 

##  Results :
Successfully trained a LightGBM predictive model.
Generated machine failure probability predictions.
Developed an interactive Streamlit dashboard.
Integrated sensor and weather data for contextual analysis. 

## Future Scope :
Real-time IoT sensor integration.
Edge AI deployment.
Cloud-based monitoring.
Automated maintenance alerts.
Deep learning-based predictive models. 

## Conclusion :
This project demonstrates how machine learning can improve industrial maintenance by predicting equipment failures before they occur. Integrating IoT sensor data with contextual weather information enables more accurate predictions, helping organizations reduce downtime and optimize maintenance planning. 

## References : 
AI4I Predictive Maintenance Dataset
LightGBM Documentation
Scikit-learn Documentation
Streamlit Documentation
Pandas Documentation
Python Official Documentation

