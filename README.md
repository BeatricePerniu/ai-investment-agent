# AI Investment Agent for Coffee Shops

This Python project simulates 30 coffee shops and uses artificial intelligence techniques to decide which ones are worth investing in.

It includes financial analysis, sentiment analysis, machine learning (regression and clustering), and generates multiple visualizations.

## Technologies Used

- Python 3.13
- pandas, numpy – data processing
- scikit-learn – Linear Regression & KMeans Clustering
- textblob – review sentiment analysis
- matplotlib – data visualizations

## Features

Simulates 30 coffee shops with:
- Monthly revenue
- Rent and staff costs
- Random review texts and average ratings

For each coffee shop:
- Calculates actual profit
- Analyzes review sentiment
- Predicts profit using linear regression
- Clusters businesses using KMeans (based on revenue and review score)
- Makes an investment decision based on profit and sentiment thresholds

Generates 3 key charts:
- Bar chart: Monthly Profit per Coffee Shop
- Scatter plot: Coffee Shop Clusters
- Pie chart: Investment Decisions

## How to Run the Project

Step 1: Install required libraries

If you're using a virtual environment:
pip install pandas matplotlib numpy scikit-learn textblob
python -m textblob.download_corpora

Step 2: Run the script

Open `ai_cafenele_agent.py` in Visual Studio Code or terminal and run:
python ai_cafenele_agent.py

The program will print a summary table and display the graphs.

## Notes

- Data is synthetically generated for reproducibility.
- Random seed is fixed using `np.random.seed(42)` and `random.seed(42)` for consistent results every run.

## Output Example

Name        Monthly_Revenue     Rent_Cost   Staff_Cost    Review_Sentiment  Profit   Invest
Kava Brew        15345           2345        4650           0.75             8350     YES
Latte Lane       10895           3280        5690           -0.25            1925     NO

## Author

This project was created as part of an academic assignment to demonstrate the use of AI agents in financial decision-making.
