# Sentiment Analysis using Sentiment 140 Dataset

## Overview
This project performs sentiment analysis on the **Sentiment 140 dataset** using both **supervised** and **unsupervised learning** techniques. The goal is to classify tweets as either **positive** or **negative** based on their content. A total of **8 classifiers** are used to explore different machine learning approaches, achieving high accuracy for sentiment prediction.

---

## File Details
### `Sentiment140.ipynb`
A comprehensive Jupyter Notebook that includes:
- **Data loading** and preprocessing
- **Feature extraction** using various techniques
- **Model training** using 8 classifiers
- **Evaluation** and comparison of the classifiers
- **Conclusions** based on model performance

---

## Dataset
**Sentiment 140**: This dataset contains **1.6 million labeled tweets**. Each tweet is categorized as either **positive (4)** or **negative (0)**, based on the sentiment expressed. Key columns in the dataset include:
- **Tweet ID**
- **Date of tweet**
- **Tweet content**
- **Sentiment label**: `0 = negative`, `4 = positive`

---

## Data Preprocessing
Preprocessing is a crucial step to convert raw text into meaningful features. Key steps involved:

### Data Cleaning:
- Remove irrelevant characters (hashtags, special symbols, hyperlinks, mentions)
- Convert all text to **lowercase**

### Text Processing:
- **Tokenization**: Split text into words (tokens)
- **Stopword Removal**: Remove common but irrelevant words (e.g., "the", "is")
- **Stemming/Lemmatization**: Reduce words to their root form

### Feature Extraction:
- **Bag of Words (BoW)** or **TF-IDF Vectorization**: Convert text data into numerical vectors for model input.

---

## Classifiers Used
To improve sentiment prediction, the following **8 classifiers** are trained and compared:

1. **Logistic Regression**: A basic classifier for binary classification tasks.
2. **Support Vector Machine (SVM)**: Maximizes the margin between classes for better generalization.
3. **Random Forest**: An ensemble of decision trees that reduces overfitting.
4. **Naive Bayes**: Suitable for text classification tasks using probability.
5. **K-Nearest Neighbors (KNN)**: Classifies based on the closest neighboring data points.
6. **Decision Trees**: Splits data into nodes based on feature importance.
7. **XGBoost**: A high-performance gradient boosting algorithm known for speed and accuracy.
8. **Gradient Boosting**: Combines weak learners to improve predictive performance.

---

## Technologies Used

### Languages:
- **Python**: The programming language used for the entire process.

### Environment:
- **Jupyter Notebook**: For an interactive coding and analysis environment.

### Libraries:
- **Pandas**: Data cleaning and manipulation.
- **NumPy**: Efficient numerical computing.
- **Scikit-learn**: Machine learning and model evaluation.
- **NLTK/Spacy**: Natural Language Processing (NLP) for tokenization, stopword removal, and stemming.
- **Matplotlib/Seaborn**: For visualizing data distributions, performance metrics, and classifier results.

---

## Steps in the Process

### Data Loading:
The **Sentiment 140** dataset is loaded into a **Pandas DataFrame** for exploration.

### Data Preprocessing:
- **Cleaning**: Remove unwanted characters and standardize the text.
- **Feature Extraction**: Use **TF-IDF** or **Bag of Words** to convert the cleaned text into vectors.

### Model Training:
- Each of the 8 classifiers is trained on the vectorized data.
- **Hyperparameters** are tuned for better performance.

### Model Evaluation:
- **Accuracy**, **Precision**, **Recall**, **F1 Score**: These metrics are calculated for each classifier.
- **Confusion Matrix**: Used to visualize classifier performance in detail.
- **ROC Curve & AUC**: Plots are generated to assess classifier sensitivity and accuracy.

### Results Comparison:
- All classifiers are compared based on their performance metrics.
- A **table of results** is created to identify the most effective model.
- Conclusion on the best classifier based on **accuracy** and other key metrics.

---

## Usage Instructions

### Clone the repository:
```bash
git clone https://github.com/yativarshney/Sentiment-analysis


