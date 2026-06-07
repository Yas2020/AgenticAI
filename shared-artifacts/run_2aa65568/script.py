ARTIFACT_DIR = '/app/artifacts/run_2aa65568'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Historical cloud revenue data
# Note: The data is in billions as per the context
historical_revenue = pd.Series({
    '2019': 54.5 / (1 + 0.29)**4,  # Back-calculated assuming 29% growth to 2023
    '2020': 54.5 / (1 + 0.29)**3,
    '2021': 54.5 / (1 + 0.29)**2,
    '2022': 54.5 / (1 + 0.29),
    '2023': 54.5
})

# Calculate the compound annual growth rate (CAGR)
years = len(historical_revenue)
start_value = historical_revenue.iloc[0]
end_value = historical_revenue.iloc[-1]
cagr = (end_value / start_value)**(1/(years-1)) - 1

# Forecast future growth for the next 5 years
forecast_years = 5
forecast_revenue = [end_value * (1 + cagr)**i for i in range(1, forecast_years + 1)]
forecast_index = range(2024, 2024 + forecast_years)
forecast_series = pd.Series(forecast_revenue, index=forecast_index)

# Plot historical and forecasted revenue
plt.figure(figsize=(10, 6))
plt.plot(historical_revenue.index, historical_revenue.values, label='Historical Revenue', marker='o')
plt.plot(forecast_series.index, forecast_series.values, label='Forecasted Revenue', marker='o', linestyle='--')
plt.title('Microsoft Cloud Revenue: Historical and Forecasted')
plt.xlabel('Year')
plt.ylabel('Revenue (Billions)')
plt.legend()
plt.grid(True)
plt.savefig(ARTIFACT_DIR + '/plot.png')

# Prepare summary
summary = {
    'historical_revenue': historical_revenue.to_dict(),
    'forecasted_revenue': forecast_series.to_dict(),
    'cagr': cagr,
    'discrepancy_note': 'CAGR calculated based on back-calculated historical data assuming 29% growth to 2023.'
}

import json
# Ensure all numpy types or decimals are converted to standard floats/ints first
print(json.dumps(summary, indent=2))
