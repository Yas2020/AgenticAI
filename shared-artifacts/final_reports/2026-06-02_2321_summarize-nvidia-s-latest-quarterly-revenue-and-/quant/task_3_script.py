ARTIFACT_DIR = '/app/artifacts/run_794ab378'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import json

# Historical revenue data from the context
# Revenue in billions
historical_revenue_data = {
    'Q4 2023': 6.05,  # From research context
    'Q4 2026': 68.1,  # From research context
    'Q1 2027': 81.6   # From research context
}

# Convert the dictionary to a DataFrame
revenue_df = pd.DataFrame(list(historical_revenue_data.items()), columns=['Quarter', 'Revenue'])

# Extract the year and quarter for numerical analysis
revenue_df['Year'] = revenue_df['Quarter'].apply(lambda x: int(x.split()[1]))
revenue_df['Quarter_Num'] = revenue_df['Quarter'].apply(lambda x: 4 if 'Q4' in x else 1)

# Create a numerical representation of time for regression
revenue_df['Time'] = revenue_df['Year'] + (revenue_df['Quarter_Num'] - 1) / 4

# Prepare the data for linear regression
X = revenue_df['Time'].values.reshape(-1, 1)
y = revenue_df['Revenue'].values

# Fit a linear regression model
model = LinearRegression()
model.fit(X, y)

# Predict future revenue for the next 4 quarters
future_times = np.array([2027 + i/4 for i in range(1, 5)])
future_revenue_predictions = model.predict(future_times.reshape(-1, 1))

# Plot the historical and predicted revenue
plt.figure(figsize=(10, 6))
plt.plot(revenue_df['Time'], revenue_df['Revenue'], label='Historical Revenue', marker='o')
plt.plot(future_times, future_revenue_predictions, label='Predicted Revenue', linestyle='--', marker='x')
plt.title('NVIDIA Revenue Growth Trend')
plt.xlabel('Year')
plt.ylabel('Revenue (Billions)')
plt.legend()
plt.grid(True)
plt.savefig(ARTIFACT_DIR + '/plot.png')

# Prepare the summary
summary = {
    'historical_revenue': historical_revenue_data,
    'predicted_revenue': dict(zip([f'Q{i} 2027' for i in range(2, 6)], future_revenue_predictions.tolist())),
    'discrepancy_note': 'Revenue for Q4 2023 is significantly lower than subsequent quarters, indicating rapid growth.'
}

# Ensure all numpy types or decimals are converted to standard floats/ints first
print(json.dumps(summary, indent=2))
