import requests
from bs4 import BeautifulSoup
import pymongo
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import variation
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# MongoDB connection setup
try:
    client = pymongo.MongoClient("mongodb+srv://saatvikmittra1:simba009@cluster0.u8nsf.mongodb.net/")
    client.admin.command('ping')
    print("MongoDB connection successful!")
except Exception as e:
    print("MongoDB connection failed:", e)

db = client["web_scraping"]
collection = db["stock_collection"]

# Function to calculate basic statistics using numpy
def analyze_stock_statistics(df):
    prices = df['price'].values
    print("\nStatistical Analysis of Stock Prices:")
    print(f"Mean Price: {np.mean(prices):.2f}")
    print(f"Median Price: {np.median(prices):.2f}")
    print(f"Standard Deviation: {np.std(prices):.2f}")
    print(f"Minimum Price: {np.min(prices):.2f}")
    print(f"Maximum Price: {np.max(prices):.2f}")
    print(f"Price Range: {np.ptp(prices):.2f}")
    print(f"Coefficient of Variation: {variation(prices):.2f}")

# Function to scrape stock data
def scrape_stock_data(symbol):
    url = f"https://finance.yahoo.com/quote/{symbol}"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    try:
        # Adjusted code to fetch price data; check Yahoo Finance to confirm
        price_tag = soup.find('fin-streamer', {'data-field': 'regularMarketPrice'})
        if price_tag:
            price = float(price_tag.text.replace(',', ''))
            date = datetime.datetime.now()                                                                                                                                                  
            # Insert data into MongoDB
            collection.insert_one({"symbol": symbol, "price": price, "date": date})
            print(f"Data for {symbol} inserted successfully.")
        else:
            print(f"Failed to locate price tag for {symbol}.")
    except AttributeError as e:
        print(f"Failed to scrape data for {symbol}. Check HTML structure. Error: {e}")

# Function to load data from MongoDB
def load_data(symbol):
    data = list(collection.find({"symbol": symbol}).sort("date", 1))
    df = pd.DataFrame(data)
    return df

# Function to plot stock data
def plot_stock_data(df):
    df['moving_avg_20'] = df['price'].rolling(window=20).mean()
    
    plt.figure(figsize=(12, 6))
    plt.plot(df['date'], df['price'], label='Price')
    plt.title("Stock Price ")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.show()

# Function to calculate volatility
def calculate_volatility(df):
    df['daily_return'] = df['price'].pct_change()
    volatility = variation(df['daily_return'].dropna())
    print("Volatility:", volatility)

# Function to train and predict using a linear regression model
def train_predictive_model(df):
    df['date_ordinal'] = pd.to_datetime(df['date']).map(datetime.datetime.toordinal)
    
    X = df[['date_ordinal']].values
    y = df['price'].values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Predict the next day’s price (single future point)
    next_day = df['date'].max() + pd.Timedelta(days=1)
    next_day_ordinal = next_day.toordinal()
    next_day_pred = model.predict([[next_day_ordinal]])
    
    # Plot actual data
    plt.figure(figsize=(12, 6))
    plt.plot(df['date'], df['price'], label='Actual Price')
    
    # Plot predicted next-day price as a single red dot
    plt.scatter([next_day], [next_day_pred[0]], color='red', label='Predicted Next Day Price')
    
    # Customize and display the plot
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.title("Actual Stock Prices with Next Day Prediction")
    plt.show()
    
    print(f"Predicted price for the next day ({next_day.date()}): {next_day_pred[0]}")
    
    return model


# Function to run the full stock analysis
def run_stock_analysis(symbol):
    print(f"Scraping data for {symbol}...")
    scrape_stock_data(symbol)
    
    print(f"Loading data for {symbol}...")
    df = load_data(symbol)
    if df.empty:
        print("No data found for the specified symbol.")
        return
    
    print(f"Plotting stock data for {symbol}...")
    plot_stock_data(df)
    
    print(f"Calculating volatility for {symbol}...")
    calculate_volatility(df)
    
    print("Analyzing statistical information for the stock...")
    analyze_stock_statistics(df)
    
    print(f"Training predictive model for {symbol}...")
    train_predictive_model(df)

# Example usage
if __name__ == "__main__":
    stockname=input("Enter Name Of The Stock ->")
    print()
    run_stock_analysis(stockname)
