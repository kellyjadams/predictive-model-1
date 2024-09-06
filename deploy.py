# import libraries
import pandas as pd
import os
from google.cloud import bigquery
import joblib
from datetime import datetime, timedelta

# Set variables
base_path = 'C:\\Users\\MyUser\\Projects\\'
model_path = os.path.join(base_path, 'model_rf.pkl')
results_path = os.path.join(base_path, 'predictions.csv')

# Set up BigQuery client
client = bigquery.Client()

# Calculate the date range for the most recent complete month
today = datetime.today()
first_of_this_month = today.replace(day=1)
last_month = first_of_this_month - timedelta(days=1)
start_date = last_month.replace(day=1).strftime('%Y-%m-%d')
end_date = last_month.strftime('%Y-%m-%d')

# Define the SQL query to fetch prediction data
query = """
SELECT
    user_id,
    DATE_TRUNC(last_login_date, MONTH) AS month,
    COUNT(DISTINCT session_id) AS num_logins,
    SUM(case when activity_type = 'game_play' then 1 else 0 end) AS games_played,
    SUM(case when activity_type = 'purchase' then amount else 0 end) AS total_purchases,
    MIN(last_login_date) AS first_login_date,
    MAX(last_login_date) AS last_login_date
FROM
    gaming.activity_log
WHERE
    DATE(last_login_date) BETWEEN @start AND @end
GROUP BY
    user_id, month
ORDER BY
    user_id, month
"""
# Set up query parameters
job_config = bigquery.QueryJobConfig(
    query_parameters=[
        bigquery.ScalarQueryParameter("start", "STRING", start_date),
        bigquery.ScalarQueryParameter("end", "STRING", end_date)
    ]
)

# Run the query
query_job = client.query(query, job_config=job_config)
df = query_job.to_dataframe()

# Clean and pivot data to wide format
df['month'] = pd.to_datetime(df['month'])
df_wide = df.pivot_table(index='user_id', columns='month', values=['num_logins', 'games_played', 'total_purchases'], aggfunc='sum')
df_wide.fillna(0, inplace=True)

# Add features
for activity in ['num_logins', 'games_played', 'total_purchases']:
    df_wide[(activity, 'trend')] = df_wide[activity].diff(axis=1).mean(axis=1)
    df_wide[(activity, 'average')] = df_wide[activity].mean(axis=1)

# Flatten the columns after wide transformation
df_wide.columns = ['_'.join(col).strip() for col in df_wide.columns.values]

# Load in the model
model = joblib.load(model_path)

# Predict using the pipeline
features = [col for col in df_wide.columns if 'trend' in col or 'average' in col]  # ensure features match training
predictions = model.predict(df_wide[features])
probabilities = model.predict_proba(df_wide[features])[:, 1]

# Save results as a dataframe
results_df = pd.DataFrame({
    'user_id': df_wide.index,
    'churn_prediction': predictions,
    'churn_probability': probabilities
})

# Save results as a CSV
results_df.to_csv(results_path, index=False)
print(f'Results saved to {results_path}')
