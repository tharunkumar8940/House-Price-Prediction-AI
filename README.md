\# 🏠 AI House Price Prediction System



An AI-powered machine learning application that predicts house prices based on property characteristics such as location, area, bedrooms, bathrooms, age, parking, floor, and distance from the city.



\## 🚀 Project Overview



The AI House Price Prediction System demonstrates an end-to-end machine learning regression workflow:



\- Data generation

\- Data preprocessing

\- Exploratory analysis

\- Feature engineering

\- Machine learning model training

\- Model comparison

\- House price prediction

\- Interactive Streamlit application

\- Model performance analysis



> \*\*Dataset Note:\*\* This project currently uses synthetic development data created for educational and portfolio demonstration purposes. The predictions should not be interpreted as real-world market valuations.



\## 🧠 Machine Learning Models



The project compares:



1\. Linear Regression

2\. Random Forest Regressor

3\. Gradient Boosting Regressor



The best-performing model is automatically selected based on R² score.



\## 📊 Features



The model uses:



\- Location

\- Area in square feet

\- Number of bedrooms

\- Number of bathrooms

\- Property age

\- Parking spaces

\- Floor

\- Total floors

\- Distance from city



\## 🛠️ Technology Stack



\- Python

\- Pandas

\- NumPy

\- Scikit-learn

\- Matplotlib

\- Seaborn

\- Plotly

\- Streamlit

\- Joblib



\## 📁 Project Structure



```text

House-Price-Prediction-AI/

│

├── app/

│   └── app.py

│

├── data/

│   ├── raw/

│   └── processed/

│

├── models/

│   ├── house\_price\_model.pkl

│   └── model\_metadata.json

│

├── notebooks/

│

├── outputs/

│   └── model\_comparison.csv

│

├── reports/

│

├── data/

│   ├── create\_dataset.py

│   └── preprocess\_data.py

│

├── train\_model.py

├── requirements.txt

├── .gitignore

└── README.md

