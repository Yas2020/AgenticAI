ARTIFACT_DIR = '/app/artifacts/run_4784f0f4'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Historical revenue data from the context
historical_revenue_data = {
    'Q3 FY2023': 18.12,  # in billions
    'Q3 FY2024': 26.0  # in billions
}

# Convert the historical data into a DataFrame
revenue_df = pd.DataFrame(list(historical_revenue_data.items()), columns=['Quarter', 'Revenue'])
revenue_df['Revenue'] = revenue_df['Revenue'].astype(float)

# Extracting the quarters as numerical values for regression
revenue_df['Quarter_Num'] = np.arange(len(revenue_df))

# Prepare the data for linear regression
X = revenue_df['Quarter_Num'].values.reshape(-1, 1)
y = revenue_df['Revenue'].values

# Fit the linear regression model
model = np.polyfit(X.flatten(), y, 1)

# Predict future revenue for the next 2 quarters
future_quarters = np.array([len(revenue_df), len(revenue_df) + 1])
predicted_revenue = np.polyval(model, future_quarters)

# Plotting the historical and predicted revenue
plt.figure(figsize=(10, 6))
plt.plot(revenue_df['Quarter'], revenue_df['Revenue'], marker='o', label='Historical Revenue')
plt.plot(['Q4 FY2024', 'Q1 FY2025'], predicted_revenue, marker='x', linestyle='--', label='Predicted Revenue')
plt.title('NVIDIA Revenue Growth Trend')
plt.xlabel('Quarter')
plt.ylabel('Revenue (in billions)')
plt.legend()
plt.grid(True)
plt.savefig(ARTIFACT_DIR + '/plot.png')

# Prepare the summary
summary = {
    'historical_revenue': historical_revenue_data,
    'predicted_revenue': {
        'Q4 FY2024': float(predicted_revenue[0]),
        'Q1 FY2025': float(predicted_revenue[1])
    },
    'discrepancy_note': 'The revenue for Q3 FY2024 was taken as 26 billion based on the second context data point.'
}

import json
# Ensure all numpy types or decimals are converted to standard floats/ints first
print(json.dumps(summary, indent=2))
