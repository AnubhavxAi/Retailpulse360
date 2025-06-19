
import pandas as pd
import numpy as np
import datetime
import random
from dash import Dash, dcc, html, Input, Output
import plotly.express as px
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Data generation
def generate_sales_data(n=1000):
    dates = pd.date_range(start="2023-01-01", end="2024-01-01", freq="D")
    products = [f"P{str(i).zfill(3)}" for i in range(10)]
    categories = ["Electronics", "Fashion", "Home", "Beauty", "Sports"]
    data = {
        "date": np.random.choice(dates, n),
        "product_id": np.random.choice(products, n),
        "category": np.random.choice(categories, n),
        "units_sold": np.random.poisson(5, n),
        "price": np.round(np.random.uniform(10, 500, n), 2),
    }
    df = pd.DataFrame(data)
    df["revenue"] = df["units_sold"] * df["price"]
    return df

def generate_reviews(n=1000):
    sentiments = [
        "Loved it!", "Terrible experience.", "Not bad.", "Great value!", "Too pricey.",
        "Excellent build.", "Damaged package.", "Exceeded expectations.", "Average.", "Worst ever."
    ]
    df = pd.DataFrame({
        "review_text": [random.choice(sentiments) for _ in range(n)],
        "product_id": [f"P{str(random.randint(0,9)).zfill(3)}" for _ in range(n)],
        "category": [random.choice(["Electronics", "Fashion", "Home", "Beauty", "Sports"]) for _ in range(n)],
        "date": [datetime.date(2023, random.randint(1, 12), random.randint(1, 28)) for _ in range(n)]
    })
    return df

def simulate_trends(df):
    trend = df.groupby("date")["units_sold"].sum().rolling(7).mean().fillna(0)
    trend = trend + np.random.normal(0, 5, len(trend))
    df_trend = trend.reset_index()
    df_trend.columns = ["date", "trend_score"]
    return df_trend

def add_sentiment(df):
    analyzer = SentimentIntensityAnalyzer()
    df["sentiment"] = df["review_text"].apply(lambda x: analyzer.polarity_scores(x)["compound"])
    return df

# Prepare data
sales = generate_sales_data()
reviews = add_sentiment(generate_reviews())
trends = simulate_trends(sales)

# Dash app
app = Dash(__name__)
app.layout = html.Div([
    html.H2("RetailPulse 360"),
    dcc.Dropdown(
        options=[{"label": i, "value": i} for i in sales["category"].unique()],
        id="category-dropdown",
        value="Electronics"
    ),
    dcc.Graph(id="rev-graph"),
    dcc.Graph(id="sent-graph"),
    dcc.Graph(id="trend-graph"),
])

@app.callback(
    [Output("rev-graph", "figure"),
     Output("sent-graph", "figure"),
     Output("trend-graph", "figure")],
    Input("category-dropdown", "value")
)
def update(category):
    df_rev = sales[sales["category"] == category].groupby("date")["revenue"].sum().reset_index()
    df_sent = reviews[reviews["category"] == category].groupby("date")["sentiment"].mean().reset_index()
    fig1 = px.line(df_rev, x="date", y="revenue", title=f"Revenue - {category}")
    fig2 = px.line(df_sent, x="date", y="sentiment", title=f"Sentiment - {category}")
    fig3 = px.line(trends, x="date", y="trend_score", title="Simulated Google Trend Score")
    return fig1, fig2, fig3

if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port=8080)
