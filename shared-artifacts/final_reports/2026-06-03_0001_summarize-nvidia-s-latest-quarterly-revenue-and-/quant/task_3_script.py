ARTIFACT_DIR = '/app/artifacts/run_b326c199'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Historical revenue data from the context
historical_revenue_data = {
    'Q1 2025': 44100.0,  # Assumed from 85% increase to Q1 2026
    'Q1 2026': 81615.0,
    'Q2 2026': 91000.0  # Forward guidance
}

# Convert the dictionary to a pandas DataFrame
revenue_df = pd.DataFrame(list(historical_revenue_data.items()), columns=['Quarter', 'Revenue'])
revenue_df['Revenue'] = revenue_df['Revenue'].astype(float)

# Calculate the quarter-over-quarter growth rates
revenue_df['QoQ Growth Rate'] = revenue_df['Revenue'].pct_change() * 100

# Plotting the revenue trend
plt.figure(figsize=(10, 6))
plt.plot(revenue_df['Quarter'], revenue_df['Revenue'], marker='o', linestyle='-')
plt.title('NVIDIA Revenue Trend')
plt.xlabel('Quarter')
plt.ylabel('Revenue (in Billions)')
plt.grid(True)
plt.savefig(ARTIFACT_DIR + '/plot.png')

# Prepare the summary
summary = {
    'revenue_trend': revenue_df.to_dict(orient='records'),
    'discrepancy_note': 'The Q1 2025 revenue is assumed based on the 85% increase to Q1 2026 as no exact figure was provided.'
}

import json
# Ensure all numpy types or decimals are converted to standard floats/ints first
print(json.dumps(summary, indent=2))
