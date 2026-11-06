# Machine Learning Mini-Projects

This repository contains two machine learning mini-projects developed as part of the Codec Technologies certification program.

## Projects Included

### 1. Stock Price Predictor (`stock_predictor.py`)
A predictive model that forecasts the next day's closing stock price based on historical market data (Open, High, Low, Volume). 
* **Technique Used:** Linear Regression
* **Libraries:** `yfinance`, `pandas`, `scikit-learn`, `matplotlib`
* **Features:** Fetches real-time historical data from Yahoo Finance, trains the regression model, and plots an actual vs. predicted price chart.

### 2. Spam Email Classifier (`spam_classifier.py`)
A Natural Language Processing (NLP) tool built to classify text messages or emails as either "spam" or "ham" (normal).
* **Technique Used:** Multinomial Naive Bayes, Bag-of-Words (CountVectorizer)
* **Libraries:** `pandas`, `scikit-learn`
* **Features:** Uses a machine learning pipeline to vectorize text and classify it, outputting a complete precision/recall classification report.

## How to Run

1. Clone the repository:
   ```bash
   git clone [https://github.com/23981a42g9-ops/ML-Mini-Projects.git](https://github.com/23981a42g9-ops/ML-Mini-Projects.git)
   cd ML-Mini-Projects
