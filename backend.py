import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
from alpha_vantage.fundamentaldata import FundamentalData
from stocknews import StockNews
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import sqlite3
from sqlite3 import Connection
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Dictionary mappings
stock_names = {
    'GOOG': 'Alphabet Inc.',
    'AAPL': 'Apple Inc.',
    'MSFT': 'Microsoft Corporation',
    'AMZN': 'Amazon.com, Inc.',
    'TSLA': 'Tesla, Inc.',
    'NFLX': 'Netflix, Inc.',
    'NVDA': 'NVIDIA Corporation',
    'IBM': 'International Business Machines Corporation',
    'INTC': 'Intel Corporation',
    'BA': 'Boeing Company',
    'CSCO': 'Cisco Systems, Inc.',
    'PFE': 'Pfizer Inc.',
    'WMT': 'Walmart Inc.',
    'DIS': 'The Walt Disney Company',
    'JNJ': 'Johnson and Johnson',
    'PG': 'Procter and Gamble Co.',
    'META': 'Meta Platforms, Inc.',
    'MCD': 'McDonald’s Corporation',
    'PEP': 'PepsiCo, Inc.'
}

stock_logos = {
    'GOOG': 'https://logo.clearbit.com/google.com',
    'AAPL': 'https://logo.clearbit.com/apple.com',
    'MSFT': 'https://logo.clearbit.com/microsoft.com',
    'AMZN': 'https://logo.clearbit.com/amazon.com',
    'TSLA': 'https://logo.clearbit.com/tesla.com',
    'NFLX': 'https://logo.clearbit.com/netflix.com',
    'NVDA': 'https://logo.clearbit.com/nvidia.com',
    'IBM': 'https://logo.clearbit.com/ibm.com',
    'INTC': 'https://logo.clearbit.com/intel.com',
    'BA': 'https://logo.clearbit.com/boeing.com',
    'CSCO': 'https://logo.clearbit.com/cisco.com',
    'PFE': 'https://logo.clearbit.com/pfizer.com',
    'WMT': 'https://logo.clearbit.com/walmart.com',
    'DIS': 'https://logo.clearbit.com/disney.com',
    'JNJ': 'https://logo.clearbit.com/jnj.com',
    'PG': 'https://logo.clearbit.com/pg.com',
    'META': 'https://logo.clearbit.com/meta.com',
    'MCD': 'https://logo.clearbit.com/mcdonalds.com',
    'PEP': 'https://logo.clearbit.com/pepsico.com'
}

# Initialize the sentiment analyzer
sid = SentimentIntensityAnalyzer()

def get_connection() -> sqlite3.Connection:
    return sqlite3.connect('user_data.db')

def add_user(first_name, last_name, mobile_number, email, occupation, date_of_birth, username, password):
    conn = get_connection()
    
    # Check if username already exists
    cursor = conn.execute('''SELECT username FROM users WHERE username = ?''', (username,))
    if cursor.fetchone():
        print("Error: Username already exists.")
        conn.close()
        return "Username already taken. Please choose another one."
    
    # Insert new user if username does not exist
    with conn:
        conn.execute('''
            INSERT INTO users (first_name, last_name, mobile_number, email, occupation, date_of_birth, username, password)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (first_name, last_name, mobile_number, email, occupation, date_of_birth, username, password))
    
    conn.close()
    return "Registration successful!"

def authenticate_user(username, password):
    conn = get_connection()
    cursor = conn.execute('''
        SELECT * FROM users WHERE username = ? AND password = ?
    ''', (username, password))
    user = cursor.fetchone()
    conn.close()
    print(f"Authenticated user: {user}")
    return user


def add_user_content(username, stock_symbol, content):
    conn = sqlite3.connect('app.db')
    c = conn.cursor()
    c.execute('''
        INSERT INTO user_content (username, stock_symbol, content)
        VALUES (?, ?, ?)
    ''', (username, stock_symbol, content))
    conn.commit()
    conn.close()

def get_user_content(stock_symbol):
    conn = sqlite3.connect('app.db')
    c = conn.cursor()
    c.execute('''
        SELECT username, content, timestamp
        FROM user_content
        WHERE stock_symbol = ?
        ORDER BY timestamp DESC
    ''', (stock_symbol,))
    content = c.fetchall()
    conn.close()
    return content

def add_to_watchlist(user_id, stock_symbol):
    conn = get_connection()
    try:
        print(f"Adding stock {stock_symbol} to watchlist for user {user_id}")
        conn.execute('''
            INSERT INTO watchlist (user_id, stock_symbol)
            VALUES (?, ?)
        ''', (user_id, stock_symbol))
        conn.commit()
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        conn.close()


def remove_from_watchlist(user_id, stock_symbol):
    conn = get_connection()
    try:
        conn.execute('''
            DELETE FROM watchlist
            WHERE user_id = ? AND stock_symbol = ?
        ''', (user_id, stock_symbol))
        conn.commit()
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        conn.close()

def get_watchlist(user_id):
    conn = get_connection()
    try:
        cursor = conn.execute('''
            SELECT stock_symbol FROM watchlist
            WHERE user_id = ?
        ''', (user_id,))
        watchlist = cursor.fetchall()
        print(f"Fetched watchlist for user {user_id}: {watchlist}")
        return [symbol[0] for symbol in watchlist]
    except Exception as e:
        print(f"An error occurred: {e}")
        return []
    finally:
        conn.close()





def load_data(ticker, start, end):
    try:
        data = yf.download(ticker, start, end)
        data.reset_index(inplace=True)
        return data
    except Exception as e:
        return pd.DataFrame()
    
def plot_stock_data(data, stock_symbol):
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=data['Date'], y=data['Close'], mode='lines', name='Close Price', line=dict(color='cyan')))
    fig1.update_layout(
        title=f'Closing Price vs Time for {stock_names.get(stock_symbol, stock_symbol)} ({stock_symbol})',
        xaxis_title='Date',
        yaxis_title='Price',
        plot_bgcolor='black',
        paper_bgcolor='black',
        font_color='white',
        xaxis=dict(gridcolor='gray', tickfont=dict(color='white')),
        yaxis=dict(gridcolor='gray', tickfont=dict(color='white')),
        title_font=dict(color='white')
    )

    fig2 = go.Figure()
    ma50 = data['Close'].rolling(50).mean()
    ma100 = data['Close'].rolling(100).mean()
    fig2.add_trace(go.Scatter(x=data['Date'], y=data['Close'], mode='lines', name='Close Price', line=dict(color='cyan')))
    fig2.add_trace(go.Scatter(x=data['Date'], y=ma50, mode='lines', name='50-Day MA', line=dict(color='magenta')))
    fig2.add_trace(go.Scatter(x=data['Date'], y=ma100, mode='lines', name='100-Day MA', line=dict(color='yellow')))
    fig2.update_layout(
        title=f'Closing Price with Moving Averages for {stock_names.get(stock_symbol, stock_symbol)} ({stock_symbol})',
        xaxis_title='Date',
        yaxis_title='Price',
        plot_bgcolor='black',
        paper_bgcolor='black',
        font_color='white',
        xaxis=dict(gridcolor='gray', tickfont=dict(color='white')),
        yaxis=dict(gridcolor='gray', tickfont=dict(color='white')),
        title_font=dict(color='white')
    )

    return fig1, fig2


# Function to calculate MSE and MAE errors
def calculate_errors(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    return mse, mae

    
def predict_stock_prices_with_errors(data):
    X = data.index.values.reshape(-1, 1)  # Feature is time (index)
    y = data['Close'].values  # Target is the closing price

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Scale the data
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train the model
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    # Predict the stock prices
    predicted_prices = model.predict(scaler.transform(data.index.values.reshape(-1, 1)))
    
    # Calculate errors on the test set
    predicted_test_prices = model.predict(X_test_scaled)
    mse, mae = calculate_errors(y_test, predicted_test_prices)

    return predicted_prices,mse,mae

def plot_predicted_vs_actual_with_errors(data, predicted_prices, stock_symbol,mse,mae):
    fig = go.Figure()

    # Plot the actual closing prices
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], mode='lines', name='Actual Price', line=dict(color='cyan')))
    
    # Plot the predicted prices
    fig.add_trace(go.Scatter(x=data['Date'], y=predicted_prices, mode='lines', name='Predicted Price', line=dict(color='magenta')))

    fig.update_layout(
        title=f'Original vs Predicted Prices for {stock_names.get(stock_symbol, stock_symbol)} (MSE: {mse:.2f}, MAE: {mae:.2f})',
        xaxis_title='Date',
        yaxis_title='Price',
        plot_bgcolor='black',
        paper_bgcolor='black',
        font_color='white'
    )    

    return fig

def display_fundamental_data(selected_symbols):
    api_key = "5S90GTR8F8REOD8T"  # Replace with your Alpha Vantage API key
    fd = FundamentalData(key=api_key, output_format="pandas")

    for stock_symbol in selected_symbols:
        try:
            overview, _ = fd.get_company_overview(symbol=stock_symbol)
            if not overview.empty:
                st.subheader(f'Company Overview: {stock_names.get(stock_symbol, stock_symbol)} ({stock_symbol})')
                st.write(overview.T)
            else:
                st.warning(f"No fundamental data available for {stock_symbol}.")
        except Exception as e:
            st.error(f"Error fetching fundamental data for {stock_symbol}: {e}")

def display_top_news(selected_symbols):
    for stock_symbol in selected_symbols:
        st.subheader(f'Top 10 News for {stock_names.get(stock_symbol, stock_symbol)} ({stock_symbol})')
        try:
            sn = StockNews(stock_symbol, save_news=False)
            df_news = sn.read_rss()
            for i in range(min(10, len(df_news))):
                st.write(f"News {i+1}: {df_news['title'][i]}")
                st.write(df_news['published'][i])
                st.write(df_news['summary'][i])
                st.markdown("---")
        except Exception as e:
            st.error(f"Error fetching news for {stock_symbol}: {e}")

def sentiment_analysis(selected_symbols):
    st.subheader("Sentiment Analysis of Latest News")
    
    for stock_symbol in selected_symbols:
        try:
            sn = StockNews(stock_symbol, save_news=False)
            df_news = sn.read_rss()

            st.write(f"Sentiment Analysis for {stock_names.get(stock_symbol, stock_symbol)} ({stock_symbol})")

            positive = 0
            neutral = 0
            negative = 0

            for i in range(min(10, len(df_news))):
                sentiment_score = sid.polarity_scores(df_news['summary'][i])
                if sentiment_score['compound'] >= 0.05:
                    positive += 1
                elif sentiment_score['compound'] <= -0.05:
                    negative += 1
                else:
                    neutral += 1

            st.write(f"Positive: {positive}")
            st.write(f"Neutral: {neutral}")
            st.write(f"Negative: {negative}")
            st.markdown("---")
        except Exception as e:
            st.error(f"Error fetching sentiment analysis for {stock_symbol}: {e}")


    