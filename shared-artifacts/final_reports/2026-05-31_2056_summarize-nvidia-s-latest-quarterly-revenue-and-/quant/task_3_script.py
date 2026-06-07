ARTIFACT_DIR = '/app/artifacts/run_43c1b5c4'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Historical revenue data from the context
historical_revenue_data = {
    '2023': 26000.0,  # in millions
    '2026_Q1': 81615.0  # in millions
}

# Convert historical data to a DataFrame
historical_revenue_df = pd.DataFrame(list(historical_revenue_data.items()), columns=['Year', 'Revenue'])

# Convert revenue to billions for consistency
historical_revenue_df['Revenue'] = historical_revenue_df['Revenue'] / 1000

# Prepare data for linear regression
X = np.array([2023, 2026]).reshape(-1, 1)  # Years
Y = historical_revenue_df['Revenue'].values  # Revenue in billions

# Calculate growth rates
revenue_2023 = historical_revenue_data['2023'] / 1000  # Convert to billions
revenue_2026_Q1 = historical_revenue_data['2026_Q1'] / 1000  # Convert to billions

# Calculate annualized growth rate from 2023 to 2026_Q1
annualized_growth_rate = ((revenue_2026_Q1 / revenue_2023) ** (1/3)) - 1

# Predict future revenue for 2027 using the growth rate
predicted_revenue_2027 = revenue_2026_Q1 * (1 + annualized_growth_rate)

# Plot historical and predicted revenue
plt.figure(figsize=(10, 6))
plt.plot(historical_revenue_df['Year'], historical_revenue_df['Revenue'], marker='o', label='Historical Revenue')
plt.scatter(2027, predicted_revenue_2027, color='red', label='Predicted Revenue 2027')
plt.title('NVIDIA Revenue Growth Trend')
plt.xlabel('Year')
plt.ylabel('Revenue (Billions)')
plt.legend()
plt.grid(True)
plt.savefig(ARTIFACT_DIR + '/plot.png')

# Prepare summary
summary = {
    'historical_revenue': historical_revenue_data,
    'predicted_revenue_2027': predicted_revenue_2027,
    'discrepancy_note': 'Revenue for 2026_Q1 is used as a proxy for 2026 full year due to lack of full year data.'
}

import json
# Ensure all numpy types or decimals are converted to standard floats/ints first
print(json.dumps(summary, indent=2))
