# import libraries
import pandas as pd
import numpy as np
from google.cloud import bigquery
import os

# BigQuery client
client = bigquery.Client()

# set variables 
start_date = '2023-01-01'
end_date = '2023-12-31'

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

query_job = client.query(query, job_config=job_config)
df = query_job.to_dataframe()

print(df)

# save as a CSV file
base_path = 'C:\\Users\\MyUser\\Projects\\'
filename = os.path.join(base_path, f'model_data.csv')
df.to_csv(filename, index=False)
print(f'Created new CSV File: {filename}')
