import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import sqlite3
import hashlib
import time
import numpy as np
import requests
from scipy.stats import entropy
from collections import defaultdict
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

conn = sqlite3.connect("stocks_game.db", check_same_thread=False)
c = conn.cursor()

c.execute(
    """CREATE TABLE IF NOT EXISTS users
             (username TEXT PRIMARY KEY, password TEXT, balance REAL)"""
)

c.execute(
    """CREATE TABLE IF NOT EXISTS transactions
             (username TEXT, stock TEXT, shares REAL, price REAL, type TEXT, date TEXT)"""
)

c.execute(
    """CREATE TABLE IF NOT EXISTS portfolio
             (username TEXT, stock TEXT, shares REAL)"""
)

conn.commit()

try:
    nltk.data.find("vader_lexicon")
except LookupError:
    nltk.download("vader_lexicon")


def create_user(username, password):
    hashed = hashlib.sha256(password.encode("utf-8")).hexdigest()
    try:
        c.execute(
            "INSERT INTO users (username, password, balance) VALUES (?, ?, ?)",
            (username, hashed, 1500.0),
        )
        conn.commit()
        return True
    except:
        return False


def verify_user(username, password):
    c.execute("SELECT password FROM users WHERE username = ?", (username,))
    result = c.fetchone()
    if result:
        hashed = hashlib.sha256(password.encode("utf-8")).hexdigest()
        return hashed == result[0]
    return False


def get_balance(username):
    c.execute("SELECT balance FROM users WHERE username = ?", (username,))
    return c.fetchone()[0]


def update_balance(username, amount):
    c.execute(
        "UPDATE users SET balance = balance + ? WHERE username = ?", (amount, username)
    )
    conn.commit()


def get_portfolio(username):
    c.execute("SELECT stock, shares FROM portfolio WHERE username = ?", (username,))
    return c.fetchall()


def update_portfolio(username, stock, shares, transaction_type):
    c.execute(
        "SELECT shares FROM portfolio WHERE username = ? AND stock = ?",
        (username, stock),
    )
    current_shares = c.fetchone()

    if transaction_type == "buy":
        if current_shares:
            new_shares = current_shares[0] + shares
            c.execute(
                "UPDATE portfolio SET shares = ? WHERE username = ? AND stock = ?",
                (new_shares, username, stock),
            )
        else:
            c.execute(
                "INSERT INTO portfolio (username, stock, shares) VALUES (?, ?, ?)",
                (username, stock, shares),
            )
    else:
        if current_shares[0] == shares:
            c.execute(
                "DELETE FROM portfolio WHERE username = ? AND stock = ?",
                (username, stock),
            )
        else:
            new_shares = current_shares[0] - shares
            c.execute(
                "UPDATE portfolio SET shares = ? WHERE username = ? AND stock = ?",
                (new_shares, username, stock),
            )
    conn.commit()


def record_transaction(username, stock, shares, price, transaction_type):
    c.execute(
        "INSERT INTO transactions (username, stock, shares, price, type, date) VALUES (?, ?, ?, ?, ?, ?)",
        (
            username,
            stock,
            shares,
            price,
            transaction_type,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        ),
    )
    conn.commit()


def get_stock_data(symbol, period="1y"):
    stock = yf.Ticker(symbol)
    hist = stock.history(period=period)
    return hist


def get_stock_price(stock):
    try:
        return stock.info["regularMarketPrice"]
    except (KeyError, TypeError):
        hist = stock.history(period="1d")
        if not hist.empty:
            return hist["Close"].iloc[-1]
        return None


def get_news(symbol=None, general_market=False):
    base_url = "https://newsapi.org/v2/everything"

    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)

    if general_market:
        query = "(stock market OR wall street OR nasdaq OR dow jones)"
    else:
        try:
            company = yf.Ticker(symbol).info["longName"]
            query = f"({symbol} OR {company}) AND (stock OR market OR trading)"
        except:
            query = f"{symbol} stock"

    params = {
        "q": query,
        "from": start_date.strftime("%Y-%m-%d"),
        "to": end_date.strftime("%Y-%m-%d"),
        "language": "en",
        "sortBy": "relevancy",
        "apiKey": "249c95b8017a4d0c98cc6c61b58b6f8b",
    }

    try:
        response = requests.get(base_url, params=params)
        if response.status_code == 200:
            return response.json()["articles"][:5]
        return []
    except:
        return []


def get_stock_info(symbol):
    try:
        stock = yf.Ticker(symbol)
        info = stock.info
        return {
            "sector": info.get("sector", "Unknown"),
            "marketCap": info.get("marketCap", 0),
            "industry": info.get("industry", "Unknown"),
        }
    except:
        return {"sector": "Unknown", "marketCap": 0, "industry": "Unknown"}


def calculate_diversification_metrics(portfolio):
    if not portfolio:
        return None

    total_value = 0
    sector_allocation = defaultdict(float)
    industry_allocation = defaultdict(float)
    market_caps = []
    stock_values = []

    for symbol, shares in portfolio:
        stock = yf.Ticker(symbol)
        price = get_stock_price(stock)
        if price is None:
            continue

        value = shares * price
        total_value += value
        stock_values.append(value)

        info = get_stock_info(symbol)
        sector_allocation[info["sector"]] += value
        industry_allocation[info["industry"]] += value
        market_caps.append(info["marketCap"])

    if total_value == 0:
        return None

    sector_weights = [v / total_value for v in sector_allocation.values()]
    sector_diversity = (
        1 - entropy(sector_weights) / np.log(len(sector_weights))
        if sector_weights
        else 0
    )

    industry_weights = [v / total_value for v in industry_allocation.values()]
    industry_diversity = (
        1 - entropy(industry_weights) / np.log(len(industry_weights))
        if industry_weights
        else 0
    )

    market_cap_categories = {"Large": 0, "Mid": 0, "Small": 0}
    for cap in market_caps:
        if cap > 10e9:
            market_cap_categories["Large"] += 1
        elif cap > 2e9:
            market_cap_categories["Mid"] += 1
        else:
            market_cap_categories["Small"] += 1

    cap_diversity = (
        1 - entropy(list(market_cap_categories.values())) / np.log(3)
        if market_caps
        else 0
    )

    concentration = 1 - (max(stock_values) / total_value if stock_values else 0)

    overall_score = (
        sector_diversity * 0.3
        + industry_diversity * 0.3
        + cap_diversity * 0.2
        + concentration * 0.2
    ) * 100

    return {
        "overall_score": overall_score,
        "sector_allocation": dict(sector_allocation),
        "industry_allocation": dict(industry_allocation),
        "market_cap_distribution": market_cap_categories,
        "concentration": concentration * 100,
        "sector_diversity": sector_diversity * 100,
        "industry_diversity": industry_diversity * 100,
        "cap_diversity": cap_diversity * 100,
    }


def analyze_sentiment(text):
    sia = SentimentIntensityAnalyzer()
    sentiment = sia.polarity_scores(text)
    return sentiment


def get_sentiment_label(compound_score):
    if compound_score >= 0.05:
        return "Positive", "green"
    elif compound_score <= -0.05:
        return "Negative", "red"
    else:
        return "Neutral", "gray"


def analyze_news_sentiment(news_articles):
    if not news_articles:
        return None

    sentiments = []
    for article in news_articles:
        if not article.get("title") or not article.get("description"):
            continue

        title_sentiment = analyze_sentiment(article["title"])
        desc_sentiment = analyze_sentiment(article["description"])

        compound_score = (
            title_sentiment["compound"] * 0.6 + desc_sentiment["compound"] * 0.4
        )

        sentiments.append(
            {
                "title": article["title"],
                "score": compound_score,
                "date": article["publishedAt"][:10],
            }
        )

    if not sentiments:
        return None

    avg_sentiment = np.mean([s["score"] for s in sentiments])
    sentiment_label, color = get_sentiment_label(avg_sentiment)

    return {
        "articles": sentiments,
        "average_score": avg_sentiment,
        "label": sentiment_label,
        "color": color,
    }


def main():
    st.set_page_config(page_title="Stock Trading Game", layout="wide")

    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    if not st.session_state.logged_in:
        st.title("Stock Trading Game")
        tab1, tab2 = st.tabs(["Login", "Register"])

        with tab1:
            st.header("Login")
            st.caption(
                "If you do not wish to create an account (all you have to do is pick a username and password), use 'ad' for the username and password."
            )
            st.caption(
                "Please do not make any edits to the admin account. To play the game, create your own account via the 'Register' tab!"
            )
            login_username = st.text_input("Username", key="login_username")
            login_password = st.text_input(
                "Password", type="password", key="login_password"
            )
            if st.button("Login"):
                if verify_user(login_username, login_password):
                    st.session_state.logged_in = True
                    st.session_state.username = login_username
                    st.rerun()
                else:
                    st.error("Invalid credentials")

        with tab2:
            st.header("Register")
            reg_username = st.text_input("Username", key="reg_username")
            reg_password = st.text_input(
                "Password", type="password", key="reg_password"
            )
            if st.button("Register"):
                if create_user(reg_username, reg_password):
                    st.success("Registration successful! Please login.")
                else:
                    st.error("Username already exists")

    else:
        st.sidebar.title(f"Welcome, {st.session_state.username}.")
        balance = get_balance(st.session_state.username)
        st.sidebar.metric("Current Balance", f"${balance:.2f}")

        if st.sidebar.button("Logout"):
            st.session_state.logged_in = False
            st.rerun()

        tab1, tab2, tab3, tab4 = st.tabs(
            ["Trade", "Portfolio", "Market Analysis", "Leaderboards"]
        )

        with tab1:
            st.header("Trade Stocks")
            symbol = st.text_input("Enter Stock Symbol (e.g., AAPL, GOOGL)").upper()

            if symbol:
                stock = yf.Ticker(symbol)
                current_price = get_stock_price(stock)
                if current_price is None:
                    st.error(f"Unable to fetch price for {symbol}")
                    return
                st.metric("Current Price", f"${current_price:.2f}")

                col1, col2 = st.columns(2)
                with col1:
                    shares = st.number_input(
                        "Number of shares", min_value=0.0, step=1.0
                    )
                    total_cost = shares * current_price
                    st.metric("Total Cost", f"${total_cost:.2f}")

                    if st.button("Buy"):
                        if total_cost <= balance:
                            update_balance(st.session_state.username, -total_cost)
                            update_portfolio(
                                st.session_state.username, symbol, shares, "buy"
                            )
                            record_transaction(
                                st.session_state.username,
                                symbol,
                                shares,
                                current_price,
                                "buy",
                            )
                            st.success(
                                f"Successfully bought {shares} shares of {symbol}"
                            )
                            time.sleep(2)
                            st.rerun()
                        else:
                            st.error("Insufficient funds")

                with col2:
                    portfolio = get_portfolio(st.session_state.username)
                    owned_shares = 0.0
                    for stock_symbol, shares_owned in portfolio:
                        if stock_symbol == symbol:
                            owned_shares = float(shares_owned)
                            break

                    shares_to_sell = st.number_input(
                        "Number of shares to sell",
                        min_value=0.0,
                        max_value=float(owned_shares),
                        step=1.0,
                    )
                    total_value = shares_to_sell * current_price
                    st.metric("Total Value", f"${total_value:.2f}")

                    if st.button("Sell"):
                        if shares_to_sell <= owned_shares:
                            update_balance(st.session_state.username, total_value)
                            update_portfolio(
                                st.session_state.username,
                                symbol,
                                shares_to_sell,
                                "sell",
                            )
                            record_transaction(
                                st.session_state.username,
                                symbol,
                                shares_to_sell,
                                current_price,
                                "sell",
                            )
                            st.success(
                                f"Successfully sold {shares_to_sell} shares of {symbol}"
                            )
                            time.sleep(2)
                            st.rerun()
                        else:
                            st.error("Insufficient shares")

                hist = get_stock_data(symbol)
                fig = go.Figure(
                    data=[
                        go.Candlestick(
                            x=hist.index,
                            open=hist["Open"],
                            high=hist["High"],
                            low=hist["Low"],
                            close=hist["Close"],
                        )
                    ]
                )
                fig.update_layout(title=f"{symbol} Stock Price History")
                st.plotly_chart(fig)

                st.subheader(f"Latest News for {symbol}")
                news = get_news(symbol)
                if news:
                    for article in news:
                        with st.expander(article["title"]):
                            st.write(f"**Source:** {article['source']['name']}")
                            st.write(f"**Published:** {article['publishedAt'][:10]}")
                            st.write(article["description"])
                            st.markdown(f"[Read more]({article['url']})")
                else:
                    st.info("No recent news found.")

                st.subheader("Technical Analysis")
                col1, col2 = st.columns(2)

                with col1:
                    hist["MA20"] = hist["Close"].rolling(window=20).mean()
                    hist["MA50"] = hist["Close"].rolling(window=50).mean()
                    hist["MA200"] = hist["Close"].rolling(window=200).mean()

                    fig_ma = px.line(
                        hist,
                        x=hist.index,
                        y=["Close", "MA20", "MA50", "MA200"],
                        title=f"{symbol} - Moving Averages",
                        template="plotly_white",
                    )
                    fig_ma.update_layout(yaxis_title="Price", xaxis_title="Date")
                    st.plotly_chart(fig_ma, use_container_width=True)

                    hist["Returns"] = hist["Close"].pct_change()
                    fig_returns = px.histogram(
                        hist,
                        x="Returns",
                        title=f"{symbol} - Returns Distribution",
                        template="plotly_white",
                        nbins=50,
                    )
                    st.plotly_chart(fig_returns, use_container_width=True)

                with col2:
                    fig_vol = px.bar(
                        hist,
                        x=hist.index,
                        y="Volume",
                        title=f"{symbol} - Volume Analysis",
                        template="plotly_white",
                    )
                    st.plotly_chart(fig_vol, use_container_width=True)

                    hist["Volatility"] = (
                        hist["Returns"].rolling(window=20).std() * np.sqrt(252) * 100
                    )
                    fig_vol = px.line(
                        hist,
                        x=hist.index,
                        y="Volatility",
                        title=f"{symbol} - 20-Day Rolling Volatility",
                        template="plotly_white",
                    )
                    fig_vol.update_layout(yaxis_title="Volatility (%)")
                    st.plotly_chart(fig_vol, use_container_width=True)

                if symbol and news:
                    st.subheader("News Sentiment Analysis")
                    sentiment_data = analyze_news_sentiment(news)

                    if sentiment_data:
                        col1, col2 = st.columns([1, 2])

                        with col1:
                            st.markdown(f"### Overall Sentiment")
                            st.markdown(
                                f"<h2 style='color: {sentiment_data['color']}'>{sentiment_data['label']}</h2>",
                                unsafe_allow_html=True,
                            )
                            st.metric(
                                "Sentiment Score",
                                f"{sentiment_data['average_score']:.2f}",
                            )

                        st.subheader("Article Sentiment Details")
                        for article in sentiment_data["articles"]:
                            sentiment_label, color = get_sentiment_label(
                                article["score"]
                            )
                            with st.expander(article["title"]):
                                st.markdown(f"**Date:** {article['date']}")
                                st.markdown(
                                    f"**Sentiment:** <span style='color: {color}'>{sentiment_label}</span> ({article['score']:.2f})",
                                    unsafe_allow_html=True,
                                )

        with tab2:
            st.header("Your Portfolio")
            portfolio = get_portfolio(st.session_state.username)

            if portfolio:
                portfolio_data = []
                total_value = 0

                for symbol, shares in portfolio:
                    stock = yf.Ticker(symbol)
                    current_price = get_stock_price(stock)
                    if current_price is None:
                        continue
                    value = shares * current_price
                    total_value += value
                    portfolio_data.append(
                        {
                            "Symbol": symbol,
                            "Shares": shares,
                            "Current Price": current_price,
                            "Total Value": value,
                        }
                    )

                df = pd.DataFrame(portfolio_data)
                st.metric("Total Portfolio Value", f"${total_value:.2f}")
                st.dataframe(df)

                fig = px.pie(
                    df,
                    values="Total Value",
                    names="Symbol",
                    title="Portfolio Distribution",
                )
                st.plotly_chart(fig)

                if portfolio:
                    st.subheader("Portfolio Diversification Analysis")

                    div_metrics = calculate_diversification_metrics(portfolio)
                    if div_metrics:
                        score_color = (
                            "green"
                            if div_metrics["overall_score"] > 70
                            else (
                                "orange" if div_metrics["overall_score"] > 40 else "red"
                            )
                        )
                        st.markdown(
                            f"### Overall Diversification Score: <span style='color:{score_color}'>{div_metrics['overall_score']:.1f}%</span>",
                            unsafe_allow_html=True,
                        )

                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric(
                                "Sector Diversity",
                                f"{div_metrics['sector_diversity']:.1f}%",
                            )
                        with col2:
                            st.metric(
                                "Industry Diversity",
                                f"{div_metrics['industry_diversity']:.1f}%",
                            )
                        with col3:
                            st.metric(
                                "Market Cap Diversity",
                                f"{div_metrics['cap_diversity']:.1f}%",
                            )
                        with col4:
                            st.metric(
                                "Concentration Risk",
                                f"{div_metrics['concentration']:.1f}%",
                            )

                        col1, col2 = st.columns(2)

                        with col1:
                            sector_data = pd.DataFrame(
                                list(div_metrics["sector_allocation"].items()),
                                columns=["Sector", "Value"],
                            )
                            fig_sector = px.pie(
                                sector_data,
                                values="Value",
                                names="Sector",
                                title="Sector Allocation",
                            )
                            st.plotly_chart(fig_sector, use_container_width=True)

                            market_cap_data = pd.DataFrame(
                                list(div_metrics["market_cap_distribution"].items()),
                                columns=["Category", "Count"],
                            )
                            fig_cap = px.bar(
                                market_cap_data,
                                x="Category",
                                y="Count",
                                title="Market Cap Distribution",
                            )
                            st.plotly_chart(fig_cap, use_container_width=True)

                        with col2:
                            industry_data = pd.DataFrame(
                                list(div_metrics["industry_allocation"].items()),
                                columns=["Industry", "Value"],
                            )
                            fig_industry = px.pie(
                                industry_data,
                                values="Value",
                                names="Industry",
                                title="Industry Allocation",
                            )
                            st.plotly_chart(fig_industry, use_container_width=True)

                        st.subheader("Diversification Recommendations")
                        if div_metrics["sector_diversity"] < 60:
                            st.warning(
                                "Consider investing in more sectors to reduce sector-specific risk."
                            )
                        if div_metrics["concentration"] < 60:
                            st.warning(
                                "Your portfolio is highly concentrated in a few positions."
                            )
                        if div_metrics["cap_diversity"] < 60:
                            st.warning(
                                "Consider diversifying across different market cap sizes."
                            )
                        if div_metrics["overall_score"] > 80:
                            st.success("Your portfolio shows good diversification!")

                if portfolio:
                    st.subheader("Portfolio Analytics")

                    end_date = datetime.now()
                    start_date = end_date - timedelta(days=365)
                    portfolio_history = pd.DataFrame()

                    for symbol, shares in portfolio:
                        stock = yf.Ticker(symbol)
                        hist = stock.history(start=start_date, end=end_date)
                        if not hist.empty:
                            portfolio_history[symbol] = hist["Close"] * shares

                    if not portfolio_history.empty:
                        portfolio_history["Total"] = portfolio_history.sum(axis=1)

                        col1, col2 = st.columns(2)

                        with col1:
                            fig_value = px.line(
                                portfolio_history,
                                x=portfolio_history.index,
                                y="Total",
                                title="Portfolio Value Over Time",
                                template="plotly_white",
                            )
                            fig_value.update_layout(
                                yaxis_title="Value ($)", xaxis_title="Date"
                            )
                            st.plotly_chart(fig_value, use_container_width=True)

                            portfolio_returns = portfolio_history["Total"].pct_change()
                            fig_returns = px.histogram(
                                portfolio_returns,
                                title="Portfolio Daily Returns Distribution",
                                template="plotly_white",
                                nbins=50,
                            )
                            st.plotly_chart(fig_returns, use_container_width=True)

                        with col2:
                            cumulative_returns = (1 + portfolio_returns).cumprod()
                            fig_cum = px.line(
                                cumulative_returns,
                                title="Cumulative Portfolio Returns",
                                template="plotly_white",
                            )
                            fig_cum.update_layout(
                                yaxis_title="Cumulative Return", xaxis_title="Date"
                            )
                            st.plotly_chart(fig_cum, use_container_width=True)

                            monthly_returns = portfolio_returns.groupby(
                                [
                                    portfolio_returns.index.year,
                                    portfolio_returns.index.month,
                                ]
                            ).sum()
                            try:
                                n_years = len(monthly_returns) // 12
                                if n_years >= 1:
                                    monthly_returns_matrix = monthly_returns.values[
                                        : 12 * n_years
                                    ].reshape(12, n_years)
                                    fig_heatmap = px.imshow(
                                        monthly_returns_matrix,
                                        title="Monthly Returns Heatmap",
                                        labels=dict(x="Year", y="Month"),
                                        color_continuous_scale="RdYlGn",
                                    )
                                    st.plotly_chart(
                                        fig_heatmap, use_container_width=True
                                    )
                                else:
                                    st.info(
                                        "Not enough data for monthly returns heatmap (need at least one full year)"
                                    )
                            except Exception as e:
                                st.info("Unable to generate monthly returns heatmap")

                    st.subheader("Performance Metrics")
                    metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(
                        4
                    )

                    if not portfolio_history.empty:
                        annual_return = (
                            (
                                portfolio_history["Total"].iloc[-1]
                                - portfolio_history["Total"].iloc[0]
                            )
                            / portfolio_history["Total"].iloc[0]
                        ) * 100
                        volatility = portfolio_returns.std() * np.sqrt(252) * 100
                        sharpe_ratio = (annual_return - 2.0) / volatility
                        max_drawdown = (
                            (
                                portfolio_history["Total"]
                                - portfolio_history["Total"].cummax()
                            )
                            / portfolio_history["Total"].cummax()
                        ).min() * 100

                        with metrics_col1:
                            st.metric("Annual Return", f"{annual_return:.1f}%")
                        with metrics_col2:
                            st.metric("Volatility", f"{volatility:.1f}%")
                        with metrics_col3:
                            st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
                        with metrics_col4:
                            st.metric("Max Drawdown", f"{max_drawdown:.1f}%")

                    st.subheader("Transaction History")
                    c.execute(
                        """
                        SELECT date, stock, type, shares, price, (shares * price) as total_value 
                        FROM transactions 
                        WHERE username = ? 
                        ORDER BY date DESC
                    """,
                        (st.session_state.username,),
                    )

                    transactions = c.fetchall()
                    if transactions:
                        transactions_df = pd.DataFrame(
                            transactions,
                            columns=[
                                "Date",
                                "Stock",
                                "Type",
                                "Shares",
                                "Price",
                                "Total Value",
                            ],
                        )

                        transactions_df["Date"] = pd.to_datetime(
                            transactions_df["Date"]
                        )
                        transactions_df["Type"] = transactions_df[
                            "Type"
                        ].str.capitalize()

                        def highlight_trades(row):
                            if row["Type"] == "Buy":
                                return ["background-color: rgba(0, 255, 0, 0.1)"] * len(
                                    row
                                )
                            else:
                                return ["background-color: rgba(255, 0, 0, 0.1)"] * len(
                                    row
                                )

                        styled_df = transactions_df.style.apply(
                            highlight_trades, axis=1
                        ).format(
                            {
                                "Date": lambda x: x.strftime("%Y-%m-%d %H:%M"),
                                "Shares": "{:.2f}",
                                "Price": "${:.2f}",
                                "Total Value": "${:.2f}",
                            }
                        )

                        st.dataframe(styled_df, use_container_width=True)

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            total_bought = transactions_df[
                                transactions_df["Type"] == "Buy"
                            ]["Total Value"].sum()
                            st.metric("Total Bought", f"${total_bought:,.2f}")
                        with col2:
                            total_sold = transactions_df[
                                transactions_df["Type"] == "Sell"
                            ]["Total Value"].sum()
                            st.metric("Total Sold", f"${total_sold:,.2f}")
                        with col3:
                            net_investment = total_bought - total_sold
                            st.metric("Net Investment", f"${net_investment:,.2f}")
                    else:
                        st.info("No transactions yet!")
            else:
                st.info("Your portfolio is empty. Start trading!")

        with tab3:
            st.subheader("Market News")
            col1, col2 = st.columns([1, 2])

            with col1:
                news_options = [
                    "General Market",
                    "Technology",
                    "Finance",
                    "Energy",
                    "Healthcare",
                ]
                selected_news = st.selectbox("Select News Category", news_options)

            with col2:
                if selected_news == "General Market":
                    news = get_news(general_market=True)
                else:
                    sector_etfs = {
                        "Technology": "XLK",
                        "Finance": "XLF",
                        "Energy": "XLE",
                        "Healthcare": "XLV",
                    }
                    news = get_news(sector_etfs.get(selected_news))

            if news:
                for article in news:
                    with st.expander(article["title"]):
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.write(f"**Source:** {article['source']['name']}")
                            st.write(f"**Published:** {article['publishedAt'][:10]}")
                            st.write(article["description"])
                            st.markdown(f"[Read more]({article['url']})")
                        with col2:
                            if article.get("urlToImage"):
                                st.image(
                                    article["urlToImage"], use_container_width=True
                                )
            else:
                st.info("No recent news found.")

            st.subheader("Market Calendar")
            today = datetime.now()
            calendar_data = {
                "Market Hours": "9:30 AM - 4:00 PM ET",
                "Pre-Market": "4:00 AM - 9:30 AM ET",
            }
            calendar_data_p2 = {
                "After-Hours": "4:00 PM - 8:00 PM ET",
                "Next Holiday": "Check NYSE calendar",
            }

            cols = st.columns(len(calendar_data))
            for col, (label, value) in zip(cols, calendar_data.items()):
                col.metric(label, value)
            cols = st.columns(len(calendar_data_p2))
            for col, (label, value) in zip(cols, calendar_data_p2.items()):
                col.metric(label, value)

            market_news = get_news(general_market=True)
            market_sentiment = analyze_news_sentiment(market_news)

            if market_sentiment:
                col1, col2 = st.columns([1, 1.5])

                with col1:
                    st.markdown("### Overall Market Sentiment")
                    st.markdown(
                        f"<h2 style='color: {market_sentiment['color']}'>{market_sentiment['label']}</h2>",
                        unsafe_allow_html=True,
                    )
                    st.metric(
                        "Market Sentiment Score",
                        f"{market_sentiment['average_score']:.2f}",
                    )

                with col2:
                    sentiment_counts = {
                        "Positive": sum(
                            1 for s in market_sentiment["articles"] if s["score"] > 0.05
                        ),
                        "Neutral": sum(
                            1
                            for s in market_sentiment["articles"]
                            if -0.05 <= s["score"] <= 0.05
                        ),
                        "Negative": sum(
                            1
                            for s in market_sentiment["articles"]
                            if s["score"] < -0.05
                        ),
                    }
                    fig = px.pie(
                        values=list(sentiment_counts.values()),
                        names=list(sentiment_counts.keys()),
                        title="Sentiment Distribution",
                    )
                    st.plotly_chart(fig, use_container_width=True)

            st.subheader("Trending Stocks Analysis")
            trending_stocks = [
                "AAPL",
                "GOOGL",
                "MSFT",
                "AMZN",
                "TSLA",
                "META",
                "NVDA",
                "JPM",
            ]

            trending_data = []
            for symbol in trending_stocks:
                stock = yf.Ticker(symbol)
                current_price = get_stock_price(stock)
                if current_price is None:
                    continue

                hist = stock.history(period="1mo")
                if not hist.empty:
                    month_change = (
                        (hist["Close"].iloc[-1] - hist["Close"].iloc[0])
                        / hist["Close"].iloc[0]
                    ) * 100
                    volatility = hist["Close"].pct_change().std() * 100

                    trending_data.append(
                        {
                            "Symbol": symbol,
                            "Price": current_price,
                            "Month Change %": month_change,
                            "Volatility %": volatility,
                        }
                    )

            if trending_data:
                trending_df = pd.DataFrame(trending_data)
                st.dataframe(
                    trending_df.style.format(
                        {
                            "Price": "${:.2f}",
                            "Month Change %": "{:.2f}%",
                            "Volatility %": "{:.2f}%",
                        }
                    )
                )

            st.header("Market Analysis")

            major_indices = {
                "^GSPC": "S&P 500",
                "^DJI": "Dow Jones",
                "^IXIC": "NASDAQ",
                "^RUT": "Russell 2000",
                "^VIX": "Volatility Index",
            }

            period = st.selectbox(
                "Select Time Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=0
            )

            col1, col2 = st.columns(2)

            for idx, (symbol, name) in enumerate(major_indices.items()):
                hist = get_stock_data(symbol, period=period)

                fig_price = px.line(
                    hist,
                    x=hist.index,
                    y="Close",
                    title=f"{name} - Price History",
                    template="plotly_white",
                )
                fig_price.update_layout(showlegend=False)

                fig_volume = px.bar(
                    hist,
                    x=hist.index,
                    y="Volume",
                    title=f"{name} - Volume",
                    template="plotly_white",
                )
                fig_volume.update_layout(showlegend=False)

                hist["Returns"] = hist["Close"].pct_change()

                fig_returns = px.histogram(
                    hist,
                    x="Returns",
                    title=f"{name} - Returns Distribution",
                    template="plotly_white",
                    nbins=50,
                )

                if idx % 2 == 0:
                    with col1:
                        st.plotly_chart(fig_price, use_container_width=True)
                        st.plotly_chart(fig_volume, use_container_width=True)
                        st.plotly_chart(fig_returns, use_container_width=True)
                else:
                    with col2:
                        st.plotly_chart(fig_price, use_container_width=True)
                        st.plotly_chart(fig_volume, use_container_width=True)
                        st.plotly_chart(fig_returns, use_container_width=True)

            st.subheader("Sector Performance")
            sectors = [
                "XLF",
                "XLK",
                "XLV",
                "XLE",
                "XLI",
                "XLC",
                "XLY",
                "XLP",
                "XLU",
                "XLB",
                "XLRE",
            ]
            sector_names = [
                "Financials",
                "Technology",
                "Healthcare",
                "Energy",
                "Industrials",
                "Communication",
                "Consumer Discretionary",
                "Consumer Staples",
                "Utilities",
                "Materials",
                "Real Estate",
            ]

            sector_data = []
            for symbol, name in zip(sectors, sector_names):
                stock = yf.Ticker(symbol)
                try:
                    hist = stock.history(period="1mo")
                    change = (
                        (hist["Close"].iloc[-1] - hist["Close"].iloc[0])
                        / hist["Close"].iloc[0]
                    ) * 100
                    sector_data.append({"Sector": name, "Change": change})
                except:
                    continue

            if sector_data:
                sector_df = pd.DataFrame(sector_data)
                fig_sectors = px.bar(
                    sector_df,
                    x="Sector",
                    y="Change",
                    title="30-Day Sector Performance (%)",
                    template="plotly_white",
                )
                st.plotly_chart(fig_sectors, use_container_width=True)

        with tab4:
            st.header("Leaderboards")

            user_portfolio = get_portfolio(st.session_state.username)
            user_portfolio_value = 0
            for symbol, shares in user_portfolio:
                stock = yf.Ticker(symbol)
                current_price = get_stock_price(stock)
                if current_price is not None:
                    user_portfolio_value += shares * current_price

            user_balance = get_balance(st.session_state.username)
            user_net_worth = user_portfolio_value + user_balance

            def get_leaderboard_data():
                c.execute("SELECT username, balance FROM users")
                users_data = c.fetchall()

                leaderboard_data = []
                for username, balance in users_data:
                    portfolio = get_portfolio(username)
                    portfolio_value = 0.0
                    total_shares = 0.0

                    for symbol, shares in portfolio:
                        stock = yf.Ticker(symbol)
                        current_price = get_stock_price(stock)
                        if current_price is not None:
                            portfolio_value += shares * current_price
                        total_shares += float(shares)

                    net_worth = portfolio_value + balance
                    leaderboard_data.append(
                        {
                            "username": username,
                            "net_worth": float(net_worth),
                            "portfolio_value": float(portfolio_value),
                            "total_shares": float(total_shares),
                        }
                    )
                return leaderboard_data

            def display_leaderboard(data, metric, metric_name):
                sorted_data = sorted(
                    data,
                    key=lambda x: (
                        x[metric] if x[metric] is not None else 0.0,
                        x["username"],
                    ),
                    reverse=True,
                )

                user_value = next(
                    item[metric]
                    for item in sorted_data
                    if item["username"] == st.session_state.username
                )
                user_rank = 1
                for i, item in enumerate(sorted_data):
                    if item["username"] == st.session_state.username:
                        user_rank = i + 1
                        break

                top_10 = sorted_data[:10]

                next_better = None
                if user_rank > 1:
                    next_better = sorted_data[user_rank - 2]

                st.subheader(f"{metric_name} Leaderboard")

                df = pd.DataFrame(top_10)
                df["Rank"] = range(1, len(df) + 1)
                df = df[["Rank", "username", metric]]

                df.columns = ["Rank", "Username", metric_name]

                if metric in ["net_worth", "portfolio_value"]:
                    df[metric_name] = df[metric_name].apply(lambda x: f"${x:,.2f}")
                else:
                    df[metric_name] = df[metric_name].apply(lambda x: f"{x:,.0f}")

                def highlight_user(row):
                    if row["Username"] == st.session_state.username:
                        return ["background-color: rgba(0, 255, 0, 0.1)"] * len(row)
                    return ["background-color: transparent"] * len(row)

                st.dataframe(
                    df.style.apply(highlight_user, axis=1), use_container_width=True
                )

                if user_rank > 10:
                    st.info(f"Your rank: #{user_rank}")

                if next_better:
                    diff = next_better[metric] - user_value
                    if metric in ["net_worth", "portfolio_value"]:
                        st.info(
                            f"${diff+0.01:,.2f} needed to reach rank #{user_rank - 1}"
                        )
                    else:
                        st.info(
                            f"{diff+1:,.0f} more shares needed to reach rank #{user_rank - 1}"
                        )

            leaderboard_data = get_leaderboard_data()

            tab1, tab2, tab3 = st.tabs(["Net Worth", "Portfolio Value", "Total Shares"])

            with tab1:
                display_leaderboard(leaderboard_data, "net_worth", "Net Worth")

            with tab2:
                display_leaderboard(
                    leaderboard_data, "portfolio_value", "Portfolio Value"
                )

            with tab3:
                st.caption("because why not.")
                display_leaderboard(leaderboard_data, "total_shares", "Total Shares")


if __name__ == "__main__":
    main()
