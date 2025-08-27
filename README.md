🕵️ Fake Job Posting Detection
📌 Overview

This project uses Natural Language Processing (NLP) and Machine Learning to detect fake job postings.
By analyzing job descriptions with TF-IDF vectorization and training a Random Forest Classifier, the model can classify postings as real or fake with ~93% accuracy.

📂 Dataset

Dataset: Fake Job Postings Dataset (Kaggle)

Contains job descriptions labeled as real (0) or fake (1).

⚙️ Tech Stack

Python 3

pandas, numpy, matplotlib, seaborn → Data Analysis & Visualization

scikit-learn → ML model (Random Forest, TF-IDF Vectorizer)

NLTK → Text Preprocessing (stopwords, stemming)

joblib → Saving & Loading Model

🚀 Steps in the Project

Data Cleaning & Preprocessing

Removed stopwords, punctuation, and applied stemming

Converted text into numerical features using TF-IDF

Model Training

Trained Random Forest Classifier

Achieved 93% accuracy

Model Saving

Saved trained model (model.pkl)

Saved vectorizer (vectorizer.pkl)

Deployment (Optional)

Can be deployed using Streamlit or Flask + Vercel/Render

📊 Results

Accuracy: ~93%

Algorithm Used: Random Forest

Feature Extraction: TF-IDF
