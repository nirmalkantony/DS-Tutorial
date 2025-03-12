import pandas as pd
import statsmodels.api as sm

# Load the dataset
data = pd.read_csv('Advertising.csv')

# Select the predictors and the target variable
predictors = data[['TV', 'radio', 'newspaper']]
target = data['sales']

# Add a constant term to the predictors for the intercept
predictors = sm.add_constant(predictors)

# Create and fit the linear regression model
model = sm.OLS(target, predictors).fit()

# Display the detailed summary of the regression model
print(model.summary())

# Extract key metrics from the model
residual_std_error = model.scale**0.5  # Residual standard error
r_squared = model.rsquared  # R-squared value
f_statistic = model.fvalue  # F-statistic

# Print the extracted metrics
print(f"Residual Standard Error (RSE): {residual_std_error}")
print(f"R-squared (R²): {r_squared}")
print(f"F-statistic: {f_statistic}")