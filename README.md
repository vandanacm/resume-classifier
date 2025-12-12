# Resume Classifier

This project implements a Resume Classifier using various machine learning and deep learning techniques. It classifies resumes into 24 different categories (e.g., HR, Designer, Information-Technology).

## Project Overview

The core analysis and modeling are contained in the Jupyter Notebook `resume-classifier.ipynb`. The project involves:
- **Exploratory Data Analysis (EDA):** Analyzing the distribution of resume categories and visualizing text data.
- **Data Preprocessing:** Cleaning resume text, removing stopwords, and performing lemmatization.
- **Feature Extraction:** Using TF-IDF and other techniques.
- **Model Training:** Training various models including:
    - Naive Bayes
    - Linear SVC
    - XGBoost
    - Transformer-based models (fine-tuning)
- **Evaluation:** Evaluating models using classification reports, confusion matrices, and ROC-AUC curves.

## Dataset

The dataset used is `Resume.csv`. It contains 2484 resumes with the following columns:
- `ID`: Unique identifier for the resume.
- `Resume_str`: The resume text in string format.
- `Resume_html`: The resume data in HTML format (dropped during analysis).
- `Category`: The target label (category of the resume).

## Dependencies

The project requires the following Python libraries:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- xgboost
- nltk
- textblob
- wordcloud
- emoji
- tqdm
- transformers
- datasets
- tf-keras

You can install the dependencies using pip:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost nltk textblob wordcloud emoji tqdm transformers datasets tf-keras
```

## Usage

1.  Ensure you have the `Resume.csv` dataset in the correct location (or update the path in the notebook).
2.  Install the required dependencies.
3.  Open `resume-classifier.ipynb` in your preferred environment (e.g., Jupyter Notebook, Google Colab, VS Code).
4.  Run the cells to reproduce the analysis and model training.
