import streamlit as st
from backend import (
    authenticate_user, plot_stock_data, display_fundamental_data, display_top_news,
    sentiment_analysis, add_user, predict_stock_prices_with_errors, stock_logos, plot_predicted_vs_actual_with_errors,
    add_user_content, get_user_content, load_data,add_to_watchlist, remove_from_watchlist, get_watchlist,stock_names
    )
from datetime import datetime
from datetime import datetime
# Define the custom CSS
# Define the custom CSS
CUSTOM_CSS = """
[data-testid="stAppViewContainer"] {
    background-image: url("https://i.postimg.cc/528knPKQ/Pngtree-stock-market-data-k-line-1168200.png");
    background-size: cover;
    background-repeat: no-repeat;
    background-position: center;
    background-attachment: fixed;
}

[data-testid="stAppViewContainer"]::before {
    content: "";
    display: block;
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background-color: rgba(0, 0, 0, 0.4);  /* Adjust the alpha value (0.3) to set transparency */
    z-index: 0;
}

[data-testid="stMarkdownContainer"] {
    color: white;
}

[data-testid="stTable"] {
    color: white;
}
[data-testid="stToolbar"] {
    display: none;
}
"""




def load_css():
    st.markdown(f'<style>{CUSTOM_CSS}</style>', unsafe_allow_html=True)
def login_page():
    load_css()
    st.title('Stock Insight Analysis - Login')
    
    username = st.text_input('Username')
    password = st.text_input('Password', type='password')
    
    if st.button('Login'):
        user = authenticate_user(username, password)
        if user:
            st.success(f"Welcome, {user[1]}!")
            st.session_state.logged_in = True
            st.session_state.username = username
            st.session_state.user_id = user[0]
        else:
            st.error('Invalid username or password')

def registration_page():
    load_css()
    st.title('Stock Insight Analysis - Register')
    
    first_name = st.text_input('First Name')
    last_name = st.text_input('Last Name')
    mobile_number = st.text_input('Mobile Number')
    email = st.text_input('Email')
    occupation = st.text_input('Occupation')
    date_of_birth = st.date_input(
        'Date of Birth',
        min_value=datetime(1950, 1, 1),  # Start from 1950
        max_value=datetime(datetime.now().year, 12, 31)  # Up to the current year
    )
    username = st.text_input('Username')
    password = st.text_input('Password', type='password')

    if st.button('Register'):
        add_user(first_name, last_name, mobile_number, email, occupation, date_of_birth, username, password)
        st.success('User registered successfully!')
    

        
def display_stock_data(stock_symbol, start_date, end_date):
    data = load_data(stock_symbol, start_date, end_date)
    if not data.empty:
        stock_fig1, stock_fig2 = plot_stock_data(data, stock_symbol)
        st.plotly_chart(stock_fig1, use_container_width=True)
        st.plotly_chart(stock_fig2, use_container_width=True)
        # Predict prices and calculate errors (MSE and MAE)
        predicted_prices, mse, mae = predict_stock_prices_with_errors(data)

        predicted_prices,mse,mae = predict_stock_prices_with_errors(data)
        prediction_fig = plot_predicted_vs_actual_with_errors(data, predicted_prices, stock_symbol,mse,mae)
        st.plotly_chart(prediction_fig, use_container_width=True)
        # Display MSE and MAE below the chart
        
        st.write(f"**Mean Squared Error (MSE):** {mse:.4f}")
        st.write(f"**Mean Absolute Error (MAE):** {mae:.4f}")
    else:
        st.warning('No data available for the selected stock symbol.')

def dashboard_page():
    load_css()
    
    st.title('Stock Insight Analysis - Dashboard')
    
    st.sidebar.title('Navigation')
    option = st.sidebar.selectbox('Select a section', ['Stock Data', 'Fundamental Data', 'News', 'Sentiment Analysis', 'User Content' ,'Watchlist'], key='section_select')
    
    st.sidebar.header('Stock Selection')
    
    # Create a list of (display_name, symbol) tuples for stock selection
    stock_options = [f"{symbol} - {name}" for symbol, name in stock_names.items()]
    selected_stock_display = st.sidebar.selectbox('Select Stock Symbol', stock_options, key='stock_symbol')
    stock_symbol = selected_stock_display.split(" - ")[0]  # Extract the symbol from the display name
    
   # stock_symbol = st.sidebar.selectbox('Select Stock Symbol', list(stock_logos.keys()), key='stock_symbol')
    logo_url = stock_logos.get(stock_symbol)
    if logo_url:
        st.sidebar.image(logo_url, width=100)  # Adjust the width as needed

    start_date = st.sidebar.date_input('Start Date', datetime(2020, 1, 1), key='start_date')
    end_date = st.sidebar.date_input('End Date', datetime.now(), key='end_date')
    if option == 'Stock Data':
        display_stock_data(stock_symbol, start_date, end_date)

    elif option == 'Fundamental Data':
        selected_symbols = st.sidebar.multiselect('Select Stock Symbols', list(stock_logos.keys()), key='fundamental_data_symbols')
        if selected_symbols:
            display_fundamental_data(selected_symbols)
        else:
            st.warning('Please select at least one stock symbol.')
    
    elif option == 'News':
        selected_symbols = st.sidebar.multiselect('Select Stock Symbols', list(stock_logos.keys()), key='news_symbols')
        if selected_symbols:
            display_top_news(selected_symbols)
        else:
            st.warning('Please select at least one stock symbol.')
    
    elif option == 'Sentiment Analysis':
        selected_symbols = st.sidebar.multiselect('Select Stock Symbols', list(stock_logos.keys()), key='sentiment_analysis_symbols')
        if selected_symbols:
            sentiment_analysis(selected_symbols)
        else:
            st.warning('Please select at least one stock symbol.')

    elif option == 'User Content':
        st.subheader('Submit Your Review or Comment')

        username = st.text_input('Username')
        content = st.text_area('Your Review/Comment')
        
        if st.button('Submit'):
            if username and content:
                add_user_content(username, stock_symbol, content)
                st.success('Content submitted successfully!')
            else:
                st.error('Please enter both username and content.')

        st.subheader('User Reviews/Comments')
        user_content = get_user_content(stock_symbol)
        if user_content:
            for entry in user_content:
                st.write(f"**{entry[0]}** - {entry[2]}")
                st.write(f"{entry[1]}")
                st.write("---")
        else:
            st.write('No reviews or comments yet.')

    elif option == 'Watchlist':
        st.subheader("Manage Your Watchlist")

        watchlist = get_watchlist(st.session_state.user_id)
        st.write("Your Watchlist:", watchlist)

        if st.button('Add to Watchlist'):
            add_to_watchlist(st.session_state.user_id, stock_symbol)
            st.success(f"{stock_symbol} added to your watchlist!")

        if st.button('Remove from Watchlist'):
            remove_from_watchlist(st.session_state.user_id, stock_symbol)
            st.success(f"{stock_symbol} removed from your watchlist!")
def app():
    st.set_page_config(page_title="Stock Prediction App", page_icon="📈")
    
    if 'logged_in' not in st.session_state or not st.session_state.logged_in:
        st.sidebar.title('Menu')
        page = st.sidebar.selectbox('Select Page', ['Login', 'Register'])
        
        if page == 'Login':
            login_page()
        elif page == 'Register':
            registration_page()
    else:
        dashboard_page()