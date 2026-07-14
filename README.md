🔧 Contextual Predictive Maintenance is an (IoT Edge AI).

## 📌 Project Overview  :
Contextual Predictive Maintenance is an IoT Edge AI system that predicts machine failures using sensor and weather data. The project applies machine learning to estimate failure probability and provides an interactive Streamlit dashboard for visualization.
- Predict machine failures using IoT sensor data.
- Improve prediction accuracy with contextual weather information.
- Reduce unexpected equipment downtime.
- Support proactive maintenance decisions.
- Visualize predictions through an interactive dashboard. 

## 🚀 Features 
- IoT sensor data processing
- Weather data integration
- Feature engineering
- Machine Learning (LightGBM)
- Failure prediction
- Streamlit dashboard
- Model evaluation and reports

 ## 📁 Project Structure 
PREDICTIVE_MAINTENANCE/
│── data/
│── notebooks/
│── src/
│── models/
│── dashboard/
│── reports/
│── tests/
│── requirements.txt
│── README.md
│── main.py

 ## ⚙️ Installation

### Clone Repository

```bash
git clone https://github.com/your-username/DSML_PREDICTIVE_MAINTENANCE.git
cd DSML_PREDICTIVE_MAINTENANCE
```

### Create Virtual Environment

Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run Complete Pipeline

```bash
python src/data_ingestion.py
python src/preprocessing.py
python src/feature_engineering.py
python src/data_fusion.py
python src/train.py
python src/evaluate.py
python src/predict.py
```

Launch Dashboard

```bash
streamlit run dashboard/app.py
```

---

## 📦 Requirements

- Python 3.10+
- pandas
- numpy
- scikit-learn
- lightgbm
- imbalanced-learn
- matplotlib
- seaborn
- plotly
- streamlit

---

## 📊 Output

- Trained Model
- Scaler
- Predictions CSV
- Evaluation Report
- Interactive Dashboard

---

## 👨‍💻 Team

- Team Members :
  - Member 1 : Sharon Wakade 
  - Member 2 : Sakshi jamadar 
  - Member 3 : Samruddhi Naikode 
  - Member 4 : vishnu ES
  - Member 5 : Venkatesh cs
  - Member 6 : Avinash k  

---

## 🛠 Technologies Used

- Python
- Machine Learning
- LightGBM
- Pandas
- Streamlit
- Plotly
- IoT
- Edge AI

---

## 📄 License

This project is developed for academic purposes.
