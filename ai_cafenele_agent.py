import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from textblob import TextBlob
import random

# Represents a single coffee shop with financial and sentiment attributes
class CoffeeShop:
    def __init__(self, name, revenue, rent, staff_cost, review_text, review_score):
        self.name = name
        self.revenue = revenue
        self.rent = rent
        self.staff_cost = staff_cost
        self.review_text = review_text
        self.review_score = review_score
        self.sentiment = self.analyze_sentiment()  # Calculates sentiment polarity
        self.profit = self.calculate_profit()      # Calculates profit
        self.predicted_profit = None
        self.cluster = None
        self.invest = None

    # Calculates monthly profit of the coffee shop
    def calculate_profit(self):
        return self.revenue - self.rent - self.staff_cost

    # Analyzes review text to determine sentiment score
    def analyze_sentiment(self):
        return TextBlob(self.review_text).sentiment.polarity

# Agent class that makes predictions, clusters and investment decisions
class AIInvestmentAgent:
    def __init__(self, shops):
        self.shops = shops
        self.model = None

    # Trains a linear regression model to predict profit
    def train_model(self):
        X = np.array([[s.revenue, s.rent, s.staff_cost, s.sentiment] for s in self.shops])
        y = np.array([s.profit for s in self.shops])
        self.model = LinearRegression().fit(X, y)
        for s, pred in zip(self.shops, self.model.predict(X)):
            s.predicted_profit = pred

    # Makes investment decision based on profit and sentiment thresholds
    def make_investment_decisions(self, profit_threshold=3000, sentiment_threshold=0.2):
        for s in self.shops:
            s.invest = "YES" if s.profit > profit_threshold and s.sentiment > sentiment_threshold else "NO"

    # Clusters coffee shops based on revenue and review score
    def cluster_shops(self, n_clusters=3):
        X = np.array([[s.revenue, s.review_score] for s in self.shops])
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(X)
        for shop, label in zip(self.shops, labels):
            shop.cluster = label

    # Converts internal data into a DataFrame for analysis or export
    def to_dataframe(self):
        return pd.DataFrame([{
            "Name": s.name,
            "Monthly_Revenue": s.revenue,
            "Rent_Cost": s.rent,
            "Staff_Cost": s.staff_cost,
            "Avg_Review_Score": s.review_score,
            "Review_Text": s.review_text,
            "Review_Sentiment": s.sentiment,
            "Profit": s.profit,
            "Predicted_Profit": s.predicted_profit,
            "Cluster": s.cluster,
            "Invest": s.invest
        } for s in self.shops])

    # Generates charts to visualize profits, clusters and investment decisions
    def visualize(self):
        df = self.to_dataframe()

        # Bar chart showing profit for each shop
        plt.figure(figsize=(12, 5))
        plt.bar(df["Name"], df["Profit"], color="skyblue")
        plt.xticks(rotation=90)
        plt.title("Monthly Profit per Coffee Shop")
        plt.ylabel("Profit (RON)")
        plt.tight_layout()
        plt.show()

        # Scatter plot showing clustering
        plt.figure(figsize=(8, 6))
        for cluster_id in sorted(df["Cluster"].unique()):
            cluster = df[df["Cluster"] == cluster_id]
            plt.scatter(cluster["Monthly_Revenue"], cluster["Avg_Review_Score"], label=f"Cluster {cluster_id}")
        plt.xlabel("Monthly Revenue")
        plt.ylabel("Average Review Score")
        plt.title("Coffee Shops Clustering")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Pie chart showing distribution of investment decisions
        investment_counts = df["Invest"].value_counts()
        plt.figure(figsize=(5, 5))
        plt.pie(investment_counts, labels=investment_counts.index, autopct='%1.1f%%', startangle=140)
        plt.title("Investment Decision by AI Agent")
        plt.show()

# === Script entry point ===

# Random seed for reproducibility
np.random.seed(42)
random.seed(42)

# List of coffee shop names and possible review texts
names = [
    "Kava Brew", "Urban Beans", "Espresso Hub", "Café Nova", "Roast & Co", "The Beanery", "Brew Republic",
    "Steamline Café", "Ground Up", "Perk Point", "The Daily Grind", "Mocha Magic", "Latte Lane", "Drip Drop",
    "Bean Palace", "Caffeine Corner", "Java House", "Brew & Bloom", "Coffee Craze", "Bean Boulevard",
    "Grind Culture", "Sip Society", "Barista's Choice", "Brewtopia", "Crema Café", "Bean Brothers",
    "The Roasted Root", "Steamy Scenes", "Velvet Roast", "Morning Mug"
]
reviews = [
    "Great atmosphere and coffee!",
    "The coffee was average and the service slow.",
    "Absolutely amazing, best coffee shop in town!",
    "Too noisy and overpriced.",
    "Loved it! Will come again.",
    "It was fine, nothing special.",
    "Terrible service and burnt coffee."
]

# Create list of CoffeeShop instances with randomized data
coffee_shops = [
    CoffeeShop(
        name=names[i],
        revenue=np.random.randint(5000, 25000),
        rent=np.random.randint(1000, 5000),
        staff_cost=np.random.randint(2000, 8000),
        review_text=random.choice(reviews),
        review_score=np.random.uniform(2.5, 5.0)
    )
    for i in range(30)
]

# Create the AI agent and process data
agent = AIInvestmentAgent(coffee_shops)
agent.train_model()
agent.make_investment_decisions()
agent.cluster_shops()

# Display the result and plots
df = agent.to_dataframe()
print(df)
agent.visualize()
