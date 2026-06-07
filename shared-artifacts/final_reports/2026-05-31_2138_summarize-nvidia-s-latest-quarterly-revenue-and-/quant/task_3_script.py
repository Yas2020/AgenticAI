ARTIFACT_DIR = '/app/artifacts/run_f707b290'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Historical revenue data from the context
# Note: All figures are in billions
revenue_data = {
    'Q1 2026': 81.6 / 1.2,  # Calculated based on 20% increase to reach Q1 2027
    'Q1 2027': 81.6
}

# Convert the dictionary to a pandas DataFrame
revenue_df = pd.DataFrame(list(revenue_data.items()), columns=['Quarter', 'Revenue'])

# Calculate the growth rate
revenue_df['Growth Rate'] = revenue_df['Revenue'].pct_change() * 100

# Plotting the revenue trend
plt.figure(figsize=(10, 6))
plt.plot(revenue_df['Quarter'], revenue_df['Revenue'], marker='o', linestyle='-')
plt.title('NVIDIA Revenue Trend')
plt.xlabel('Quarter')
plt.ylabel('Revenue (Billions)')
plt.grid(True)
plt.savefig(ARTIFACT_DIR + '/plot.png')

# Prepare the summary
summary = {
    'revenue_trend': revenue_df.to_dict(orient='records'),
    'discrepancy_note': "The revenue for Q1 2026 was back-calculated using the 20% growth rate provided for Q1 2027."
}

import json
# Ensure all numpy types or decimals are converted to standard floats/ints first
print(json.dumps(summary, indent=2))
