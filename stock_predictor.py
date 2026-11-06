import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def predict_stock_price(ticker="RELIANCE.NS", start_date="2022-01-01", end_date="2023-12-31"):
    print(f"Fetching data for {ticker}...")
    df = yf.download(ticker, start=start_date, end=end_date)
    
    # Feature Engineering
    df['Target_Close'] = df['Close'].shift(-1)
    df.dropna(inplace=True)
    
    X = df[['Open', 'High', 'Low', 'Volume']]
    y = df['Target_Close']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Model Training
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Evaluation
    predictions = model.predict(X_test)
    print(f"MSE: {mean_squared_error(y_test, predictions):.2f}")
    print(f"R2 Score: {r2_score(y_test, predictions):.2f}")
    
    # Visualization (Great for GitHub)
    plt.figure(figsize=(10,5))
    plt.plot(y_test.values[:50], label='Actual Prices', marker='o')
    plt.plot(predictions[:50], label='Predicted Prices', marker='x')
    plt.title(f"{ticker} - Actual vs Predicted (Sample 50 days)")
    plt.legend()
    plt.savefig("stock_prediction_chart.png")
    print("Chart saved as 'stock_prediction_chart.png'")

if __name__ == "__main__":
    predict_stock_price()