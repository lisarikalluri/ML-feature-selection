from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import pandas as pd

housing_dataset = pd.read_csv('melbourne_housing_raw.csv')
missing_data_percent = housing_dataset.isnull().mean() * 100
drop_columns = [col for col in missing_data_percent.index if missing_data_percent[col] > 20 and col != 'Price']
filtered_dataset = housing_dataset.drop(columns=drop_columns)
filtered_dataset = filtered_dataset.dropna(subset=['Price'])

X_features = filtered_dataset.drop(columns=['Price', 'Date', 'Suburb', 'Type', 'Method', 'SellerG', 'CouncilArea', 'Regionname'])
y_target = filtered_dataset['Price']

numeric_cols = X_features.select_dtypes(include=[float, int])
X_features[numeric_cols.columns] = numeric_cols.fillna(numeric_cols.mean())

X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(X_features, y_target, test_size=0.2, random_state=42)

def assess_model(train_data, test_data, train_label, test_label):
    rf_model = RandomForestRegressor(random_state=42)
    rf_model.fit(train_data, train_label)
    preds = rf_model.predict(test_data)
    mse_error = mean_squared_error(test_label, preds)
    return mse_error

mse_error_value = assess_model(X_train_split, X_test_split, y_train_split, y_test_split)
print(f"Model performance after filtering columns with >20% missing values: MSE = {mse_error_value:.2f}")

