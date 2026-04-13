# 📧 Spam Email Classifier

## 📌 Overview
This project builds a machine learning model to classify emails as spam or not spam using Natural Language Processing techniques.

## 🚀 Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- TF-IDF Vectorizer
- Logistic Regression
- Streamlit

## ⚙️ Workflow
- Data Cleaning
- Text Preprocessing (tokenization, stopword removal)
- Feature Extraction using TF-IDF
- Model Training
- Evaluation

## 📊 Results
- Accuracy: ~95%
- High precision and recall

## 📂 Dataset
Spam email dataset (~6000 messages)

## ▶️ Run Project

pip install -r requirements.txt  
python src/train.py  
streamlit run app.py  

## 📁 Structure
- notebook/analysis.ipynb → EDA + visualization  
- src/train.py → training pipeline  
- app.py → UI  

## 📌 Conclusion
The model effectively classifies spam emails and can be extended for real-world filtering systems.
