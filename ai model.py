import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

# STEP 1: Load CSV
data = pd.read_csv("sample_101_IndianHouses.csv")

# STEP 2: Select columns
X = data[['Area', 'BHK', 'Bathroom', 'Locality']]
y = data['Price']

# STEP 3: Encode Locality (city name)
pre = ColumnTransformer([
    ('locality', OneHotEncoder(handle_unknown='ignore'), ['Locality'])
], remainder='passthrough')

# STEP 4: ML Model
model = Pipeline([
    ('preprocess', pre),
    ('rf', RandomForestRegressor(n_estimators=100, random_state=42))
])

# STEP 5: Train the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)

print("Model Trained Successfully!\n")

# ------------------------------- #
#       USER INPUT SECTION
# ------------------------------- #
area = float(input("Enter Area in sqft: "))
bhk = int(input("Enter number of BHK: "))
bath = int(input("Enter number of Bathrooms: "))
loc = input("Enter Locality / City Name: ")

# STEP 6: Create user input DataFrame
user_data = pd.DataFrame([{
    "Area": area,
    "BHK": bhk,
    "Bathroom": bath,
    "Locality": loc
}])

# STEP 7: Predict
price = model.predict(user_data)[0]
print("\nEstimated House Price =", int(price))
