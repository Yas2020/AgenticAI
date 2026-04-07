ARTIFACT_DIR = '/app/artifacts/run_f362f862'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Data from the context
quarters = ['Q1 2026', 'Q4 2026']
data_center_revenue = [39.1, 62.3]  # in billion dollars

overall_revenue_q1_2026 = 44.1  # in billion dollars

data_center_revenue_full_year_2026 = 193.7  # in billion dollars

# Calculate the percentage of data center revenue in Q1 2026
percentage_data_center_q1_2026 = (data_center_revenue[0] / overall_revenue_q1_2026) * 100

# Calculate the percentage increase from Q1 2026 to Q4 2026
percentage_increase_q1_to_q4 = ((data_center_revenue[1] - data_center_revenue[0]) / data_center_revenue[0]) * 100

# Create a DataFrame for plotting
revenue_df = pd.DataFrame({
    'Quarter': quarters,
    'Data Center Revenue (Billion $)': data_center_revenue
})

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(revenue_df['Quarter'], revenue_df['Data Center Revenue (Billion $)'], marker='o', linestyle='-')
plt.title("NVIDIA Data Center Revenue Growth")
plt.xlabel("Quarter")
plt.ylabel("Revenue (Billion $)")
plt.grid(True)
plt.savefig(ARTIFACT_DIR + '/plot.png')

# Summary of findings
summary = {
    "percentage_data_center_q1_2026": percentage_data_center_q1_2026,
    "percentage_increase_q1_to_q4": percentage_increase_q1_to_q4,
    "data_center_revenue_full_year_2026": data_center_revenue_full_year_2026
}

summary