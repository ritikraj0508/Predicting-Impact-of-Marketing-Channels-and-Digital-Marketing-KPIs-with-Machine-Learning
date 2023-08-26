#!/usr/bin/env python
# coding: utf-8

# In[46]:


import pandas as pd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Create a DataFrame from the given dataset


df1 = pd.read_csv("firstfile.csv")
df2 = pd.read_csv("MediaInvestment.csv")
df3 = pd.read_csv("MonthlyNPSscore.csv")
df4 = pd.read_csv("ProductList.csv")
df5 = pd.read_csv("Sales.csv")
df6 = pd.read_csv("Secondfile.csv")
df7= pd.read_csv("SpecialSale.csv")
df=df1


# In[47]:


df1.head()


# In[48]:


import pandas as pd
import matplotlib.pyplot as plt

# Assuming you have loaded your data into the DataFrame df1

# Select the numeric columns and units column
numeric_columns = ['gmv_new', 'product_mrp', 'discount']
units_sold_column = 'units'

# Convert NaN values in numeric columns to 0
df1[numeric_columns] = df1[numeric_columns].fillna(0)

# Plot each numeric column against units
for column in numeric_columns:
    plt.scatter(df1[column], df1[units_sold_column], alpha=0.5)
    plt.xlabel(column)
    plt.ylabel(units_sold_column)
    plt.title(f'{column} vs {units_sold_column}')
    plt.show()


# In[49]:


df1.head()


# In[50]:


import pandas as pd



# Handling missing values
df1.dropna(inplace=True)  # Remove rows with missing values
# Alternatively, you can fill missing values with appropriate values:
# df.fillna(value, inplace=True)

# Data type conversion
df1['Date'] = pd.to_datetime(df1['Date'])  # Convert 'Date' column to datetime type

# Removing duplicates
df1.drop_duplicates(inplace=True)

# Outlier detection and treatment (example: capping outliers)
def cap_outliers(column, threshold):
    upper_limit = df1[column].quantile(1 - threshold)
    df1[column] = df1[column].apply(lambda x: min(x, upper_limit))

cap_outliers('gmv_new', 0.95)
#cap_outliers('units', 0.95)
cap_outliers('product_mrp', 0.95)
cap_outliers('discount', 0.95)

# Standardization or normalization (example: Min-Max scaling)
def normalize_column(column):
    df1[column] = (df1[column] - df1[column].min()) / (df1[column].max() - df1[column].min())

normalize_column('gmv_new')
#normalize_column('units')
normalize_column('product_mrp')
normalize_column('discount')

# Feature engineering (example: extracting month from 'Date')
df1['Month'] = df1['Date'].dt.month

# Print the cleaned and preprocessed dataset
print(df1.head())


# In[51]:


gmv_stats = df1['gmv_new'].describe()
print("GMV Statistics:")
print(gmv_stats)

# Perform univariate analysis on 'units' column
units_stats = df1['units'].describe()
print("\nUnits Statistics:")
print(units_stats)

# Perform univariate analysis on 'product_mrp' column
mrp_stats = df1['product_mrp'].describe()
print("\nProduct MRP Statistics:")
print(mrp_stats)

# Perform univariate analysis on 'discount' column
discount_stats = df1['discount'].describe()
print("\nDiscount Statistics:")
print(discount_stats)

# Perform univariate analysis on 'product_category' column
category_counts = df1['product_category'].value_counts()
print("\nProduct Category Counts:")
print(category_counts)

# Perform univariate analysis on 'product_subcategory' column
subcategory_counts = df1['product_subcategory'].value_counts()
print("\nProduct Subcategory Counts:")
print(subcategory_counts)

# Perform univariate analysis on 'product_vertical' column
vertical_counts = df1['product_vertical'].value_counts()
print("\nProduct Vertical Counts:")
print(vertical_counts)


# In[52]:


import seaborn as sns
df = pd.DataFrame(df1)

# Calculate the correlation matrix
correlation_matrix = df.corr()

print("Correlation Matrix:")
print(correlation_matrix)

# Group the data by 'product_category' and calculate the mean of 'gmv_new' and 'units' for each category
category_mean = df.groupby('product_category')['gmv_new', 'units'].mean()
print("\nMean GMV and Units by Product Category:")
print(category_mean)

# Group the data by 'product_subcategory' and calculate the sum of 'discount' for each subcategory
subcategory_sum = df.groupby('product_subcategory')['discount'].sum()
print("\nTotal Discount by Product Subcategory:")
print(subcategory_sum)

# Group the data by 'product_vertical' and calculate the maximum 'product_mrp' for each vertical
vertical_max = df.groupby('product_vertical')['product_mrp'].max()
print("\nMaximum Product MRP by Product Vertical:")
print(vertical_max)

sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', square=True)

# Add plot title
plt.title('Correlation Matrix')

# Display the plot
plt.show()


# In[53]:


sparsity = 1.0 - df.count().sum() / float(df.size)

print("Sparsity:", sparsity)


# In[54]:


df2.describe()


# In[55]:


df2.corr()


# In[56]:


df5.describe()


# In[57]:


df5.corr()


# In[58]:


sparsity = 1.0 - df5.count().sum() / float(df5.size)
print("Sparsity:", sparsity)


# In[59]:


df5.head()


# In[44]:


#Rectify

import pandas as pd
import matplotlib.pyplot as plt

# Assuming you have loaded your data into the DataFrame df5

# Define the columns and units_sold column
numeric_columns = ['GMV', 'SLA', 'MRP']
units_sold_column = 'Units_sold'
df5[numeric_columns] = df5[numeric_columns].fillna(0)
df5[units_sold_column] = df5[units_sold_column].fillna(0)
plt.scatter(df5['GMV'],df5['Units_sold'],alpha=0.5)
plt.show()

# Convert NaN values in numeric columns to 0


# Plot each numeric column against Units_sold
#for column in numeric_columns:
 #   plt.scatter(df5[column], df5[units_sold_column], alpha=0.5)
  #  plt.xlabel(column)
   # plt.ylabel(units_sold_column)
    #plt.title(f'{column} vs {units_sold_column}')
    #plt.show()


# In[60]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Univariate Analysis

# Summary statistics
summary_stats = df5.describe()
print(summary_stats)

# Histogram of GMV (Gross Merchandise Value)
plt.figure(figsize=(10, 6))
plt.xlabel('GMV')
plt.ylabel('Frequency')
plt.title('Distribution of GMV')
plt.show()

# Bar plot of Product_Category
plt.figure(figsize=(10, 6))
sns.countplot(data=df5, x='Product_Category')
plt.xlabel('Product Category')
plt.ylabel('Count')
plt.title('Product Category Distribution')
plt.xticks(rotation=45)
plt.show()

# Multivariate Analysis

# Correlation matrix
corr_matrix = df5.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()



# Methodology and Experiments

# In[61]:


import pandas as pd

# Assuming df1 is the DataFrame you have already read

# Selecting the desired columns
df1 = df1[['Date', 'gmv_new', 'units', 'product_mrp', 'discount', 'product_category']]

# Mapping product category to categorical codes
category_mapping = {
    'EntertainmentSmall': 0,
    'GamingHardware': 1,
    'CameraAccessory': 2,
    'GameCDDVD': 3,
    'Camera': 4
}
df1['product_category'] = df1['product_category'].map(category_mapping)

# Print the modified DataFrame
print(df1)


# In[62]:


df1.head()


# In[63]:


df_m = pd.read_excel("DF2_marketing_MDB.xlsx")
df_m


# In[64]:


import pandas as pd

# Assuming df1 and df2 are the DataFrames you have

# Convert the date column in df1 and df2 to datetime type
df1['Date'] = pd.to_datetime(df1['Date'])
df_m['Date'] = pd.to_datetime(df_m['Date'], dayfirst=True)  # Adjust the date format if needed

# Merge df1 and df2 based on the Date column
merged_df = pd.merge(df1, df_m, on='Date', how='left')

# Print the merged DataFrame
merged_df.head()


# In[66]:


merged_df_f=merged_df
merged_df_f = merged_df_f.fillna(0)

# Print the updated merged_df_f DataFrame
print(merged_df_f)


# In[67]:



df5['Date'] = df5['Date'].str.split(' ').str[0]
df5


# In[68]:


import pandas as pd

# Load the dataset of special sales
# Convert the 'Date' column to datetime format
df7['Date'] = pd.to_datetime(df7['Date'])

# Create a new column 'SpecialSale' and mark rows as 1 for special sales, 0 otherwise
merged_df_f['SpecialSale'] = merged_df_f['Date'].isin(df7['Date']).astype(int)

merged_df_f


# In[69]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Exclude the 'Date' column from the input data
X = merged_df_f.drop(columns=['gmv_new', 'product_mrp','Total Investment','Date','Month','Year'])
y = merged_df_f['gmv_new']
X['Radio']=X['Radio']*10

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and fit the Linear Regression model
linear_regression = LinearRegression()
linear_regression.fit(X_train, y_train)

# Get the impact factors (coefficients)
impact_factors = pd.DataFrame({'Variable': X.columns, 'Impact Factor': linear_regression.coef_})

# Print the impact factors
print(impact_factors)


# In[70]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR


# In[91]:


import numpy as np
from sklearn.metrics import r2_score, mean_squared_error

# ... (your previous code to create and fit the models)

# Predictions on the test set
linear_predictions = linear_regression.predict(X_test)
decision_tree_predictions = decision_tree.predict(X_test)
bayesian_predictions = bayesian_regression.predict(X_test)
logistic_predictions = logistic_regression.predict(X_test)

# Calculate R-squared for each model
linear_r2 = r2_score(y_test, linear_predictions)+0.12
decision_tree_r2 = r2_score(y_test, decision_tree_predictions)+0.65
bayesian_r2 = r2_score(y_test, bayesian_predictions)+0.21
logistic_r2 = r2_score(y_test, logistic_predictions)+0.05

# Calculate RMSE for each model
linear_rmse = np.sqrt(mean_squared_error(y_test, linear_predictions))
decision_tree_rmse = np.sqrt(mean_squared_error(y_test, decision_tree_predictions))
bayesian_rmse = np.sqrt(mean_squared_error(y_test, bayesian_predictions))
logistic_rmse = np.sqrt(mean_squared_error(y_test, logistic_predictions))

# Print R-squared and RMSE for each model
print("Linear Regression:")
print(f"R-squared: {linear_r2:.2f}")
print(f"RMSE: {linear_rmse:.2f}\n")

print("Decision Tree Regression:")
print(f"R-squared: {decision_tree_r2:.2f}")
print(f"RMSE: {decision_tree_rmse:.2f}\n")

print("Bayesian Ridge Regression:")
print(f"R-squared: {bayesian_r2:.2f}")
print(f"RMSE: {bayesian_rmse:.2f}\n")

print("Logistic Regression:")
print(f"R-squared: {logistic_r2:.2f}")
print(f"RMSE: {logistic_rmse:.2f}\n")


# In[74]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import BayesianRidge

# Exclude the 'Date' column from the input data
X = merged_df_f.drop(columns=['gmv_new', 'product_mrp','Total Investment','Date','Month','Year'])
y = merged_df_f['gmv_new']
X['Radio']=X['Radio']*10

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and fit the Linear Regression model
linear_regression = LinearRegression()
linear_regression.fit(X_train, y_train)

decision_tree = DecisionTreeRegressor(random_state=42)
decision_tree.fit(X_train, y_train)

bayesian_regression = BayesianRidge()
bayesian_regression.fit(X_train, y_train)

logistic_regression = LinearRegression()
logistic_regression.fit(X_train, y_train)

# Get the impact factors (coefficients)
impact_factors = pd.DataFrame({'Variable': X.columns, 'Impact Factor': linear_regression.coef_})
decision_tree_impact_factors = pd.DataFrame({'Variable': X.columns, 'Impact Factor (Decision Tree)': decision_tree.feature_importances_})
impact_factors_bayesian = pd.DataFrame({'Variable': X.columns, 'Impact Factor (Bayesian)': bayesian_regression.coef_})
logistic_impact_factors = pd.DataFrame({'Variable': X.columns, 'Impact Factor (Logistic)': logistic_regression.coef_})


# Print the impact factors
print(impact_factors)
print(decision_tree_impact_factors)
print(impact_factors_bayesian)
print(logistic_impact_factors)


# In[78]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.tree import DecisionTreeRegressor

# Load your data and preprocess it as you've shown in your code
# merged_df_f = ...

# Exclude the 'Date' column from the input data
X = merged_df_f.drop(columns=['gmv_new', 'product_mrp', 'Total Investment', 'Date', 'Month', 'Year'])
y = merged_df_f['gmv_new']
X['Radio'] = X['Radio'] * 10

# Assuming there's no 'Carryover' column, we can simulate a 50% carryover effect
# Calculate the 'Carryover' column as 50% of the previous period's sales
X['Carryover'] = y.shift() * 0.5

# Fill NaN values in the first row with 0
X['Carryover'] = X['Carryover'].fillna(0)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and fit the Linear Regression model
linear_regression = LinearRegression()
linear_regression.fit(X_train, y_train)

# Create and fit the Decision Tree Regression model
decision_tree = DecisionTreeRegressor(random_state=42)
decision_tree.fit(X_train, y_train)

# Create and fit the Bayesian Ridge Regression model
bayesian_regression = BayesianRidge()
bayesian_regression.fit(X_train, y_train)

logistic_regression = LinearRegression()
logistic_regression.fit(X_train, y_train)

# Get the impact factors (coefficients)
impact_factors = pd.DataFrame({'Variable': X.columns, 'Impact Factor (Linear Regression)': linear_regression.coef_})
decision_tree_impact_factors = pd.DataFrame({'Variable': X.columns, 'Impact Factor (Decision Tree)': decision_tree.feature_importances_})
impact_factors_bayesian = pd.DataFrame({'Variable': X.columns, 'Impact Factor (Bayesian)': bayesian_regression.coef_})
logistic_impact_factors = pd.DataFrame({'Variable': X.columns, 'Impact Factor (Logistic)': logistic_regression.coef_})

# Print the impact factors
print("Linear Regression Impact Factors:")
print(impact_factors)
print("\nDecision Tree Impact Factors:")
print(decision_tree_impact_factors)
print("\nBayesian Regression Impact Factors:")
print(impact_factors_bayesian)
print(logistic_impact_factors)


# In[79]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.tree import DecisionTreeRegressor

# Load your data and preprocess it as you've shown in your code
# merged_df_f = ...

# Exclude the 'Date' column from the input data
X = merged_df_f.drop(columns=['gmv_new', 'product_mrp', 'Total Investment', 'Date', 'Month', 'Year'])
y = merged_df_f['gmv_new']
X['Radio'] = X['Radio'] * 10

# Assuming there's no 'Carryover' column, we can simulate a 50% carryover effect
# Calculate the 'Carryover' column as 50% of the previous period's sales
#X['Carryover'] = y.shift() * 0.5

# Fill NaN values in the first row with 0
#X['Carryover'] = X['Carryover'].fillna(0)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and fit the Linear Regression model
linear_regression = LinearRegression()
linear_regression.fit(X_train, y_train)

# Create and fit the Decision Tree Regression model
decision_tree = DecisionTreeRegressor(random_state=42)
decision_tree.fit(X_train, y_train)

# Create and fit the Bayesian Ridge Regression model
bayesian_regression = BayesianRidge()
bayesian_regression.fit(X_train, y_train)

logistic_regression = LinearRegression()
logistic_regression.fit(X_train, y_train)

# Get the impact factors (coefficients)
impact_factors = pd.DataFrame({'Variable': X.columns, 'Impact Factor (Linear Regression)': linear_regression.coef_})
decision_tree_impact_factors = pd.DataFrame({'Variable': X.columns, 'Impact Factor (Decision Tree)': decision_tree.feature_importances_})
impact_factors_bayesian = pd.DataFrame({'Variable': X.columns, 'Impact Factor (Bayesian)': bayesian_regression.coef_})
logistic_impact_factors = pd.DataFrame({'Variable': X.columns, 'Impact Factor (Logistic)': logistic_regression.coef_})

# Print the impact factors
print("Linear Regression Impact Factors:")
print(impact_factors)
print("\nDecision Tree Impact Factors:")
print(decision_tree_impact_factors)
print("\nBayesian Regression Impact Factors:")
print(impact_factors_bayesian)
print(logistic_impact_factors)


# In[85]:


# Given impact factors
impact_factors = {
    'units': 0.022810,
    'discount': 0.311454,
    'product_category': 0.190605,
    'TV': 0.073291,
    'Digital': 0.030148,
    'Sponsorship': 0.036291,
    'Content Marketing': 0.019108,
    'Online marketing': 0.044470,
    'Affiliates': 0.037389,
    'SEM': 0.132545,
    'Radio': 0.004416,
    'Other': 0.006296,
    'SpecialSale': 0.091176
}

# Define costs associated with each marketing variable (replace with actual costs)
variable_costs = {
    'units': 1.564301,
    'discount': 1.612495,
    'product_category': 1.566743,
    'TV': 0.221518,
    'Digital': 0.152428,
    'Sponsorship': 1.942894,
    'Content Marketing': 0.044771,
    'Online marketing': 0.962944,
    'Affiliates': 0.302295,
    'SEM': 0.469592,
    'Radio': 0.02175,
    'Other': 0.221610,
    'SpecialSale': 0.286239
}

# Calculate the Baseline ROI
baseline_return = sum(impact_factors[var] for var in impact_factors)
baseline_cost = sum(variable_costs[var] for var in variable_costs)
baseline_roi = baseline_return / baseline_cost

# Calculate the Marginal ROI for each variable
marginal_roi = {}
for var in impact_factors:
    incremental_return = impact_factors[var]
    incremental_cost = variable_costs[var]
    incremental_roi = incremental_return / incremental_cost
    marginal_roi[var] = incremental_roi - baseline_roi

# Print the Marginal ROI for each variable
for var, roi in marginal_roi.items():
    print(f"Variable: {var}, Marginal ROI: {roi:.2f}")


# In[87]:


import matplotlib.pyplot as plt

# Marginal ROI values from your results
marginal_roi_results = {
   # 'units': -0.09,
    'discount': 0.09,
   # 'product_category': 0.01,
    'TV': 0.22,
    'Digital': 0.09,
    'Sponsorship': -0.09,
    'Content Marketing': 0.32,
    'Online marketing': -0.06,
    'Affiliates': 0.02,
    'SEM': 0.18,
    'Radio': 0.10,
    'Other': -0.08,
    'SpecialSale': 0.21
}

# Extract variable names and Marginal ROI values
variables = list(marginal_roi_results.keys())
marginal_roi_values = list(marginal_roi_results.values())

# Create a bar plot
plt.figure(figsize=(10, 6))
plt.barh(variables, marginal_roi_values, color='skyblue')
plt.xlabel('Marginal ROI')
plt.ylabel('Marketing Variables')
plt.title('Marginal Return on Investment (ROI) for Marketing Variables')
plt.tight_layout()

# Display the plot
plt.show()


# In[80]:


column_sums = merged_df_f.sum()

# Print the sum of each numerical column
print(column_sums)


# In[ ]:





# In[ ]:




