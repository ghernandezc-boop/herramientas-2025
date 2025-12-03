import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# Cargar datos
df = pd.read_csv(r'C:\Users\User win10pro\Documents\herramientas\herramientas-2025\aaa_2\BMW.csv')

# Codificar todo
df_encoded = df.copy()
for col in df_encoded.select_dtypes(include=['object']).columns:
    df_encoded[col] = LabelEncoder().fit_transform(df_encoded[col].astype(str))

# Features y target
X = df_encoded.drop(columns=['Price_USD'])
y = df_encoded['Price_USD']

# Modelo simple
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf = RandomForestRegressor(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)
preds = rf.predict(X_test)
mae = mean_absolute_error(y_test, preds)
print(f"MAE con RandomForest y todas las variables: ${mae:.2f}")