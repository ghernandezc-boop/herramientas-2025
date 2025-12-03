import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

# 1. Cargar datos
df = pd.read_csv(r'C:\Users\User win10pro\Documents\herramientas\herramientas-2025\aaa_2\BMW.csv')

# 2. Preprocesamiento
# Seleccionar columnas relevantes
features = ['Year', 'Engine_Size_L', 'Mileage_KM']
categorical = ['Model', 'Region', 'Fuel_Type', 'Transmission']

# Codificar columnas categóricas
for col in categorical:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))

# Variables de entrada y salida
X = df[features + categorical].values
y = df['Price_USD'].values.reshape(-1, 1)

# Escalar
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X = scaler_X.fit_transform(X)
y = scaler_y.fit_transform(y)

# 3. División y DataLoader
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# 4. Definir red neuronal
class Net(nn.Module):
    def __init__(self, input_dim):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)

model = Net(X_train.shape[1])
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 5. Entrenamiento
epochs = 100
for epoch in range(epochs):
    for xb, yb in train_loader:
        pred = model(xb)
        loss = criterion(pred, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# 6. Evaluación
model.eval()
with torch.no_grad():
    preds = model(X_test)
    preds = scaler_y.inverse_transform(preds.numpy())
    y_true = scaler_y.inverse_transform(y_test.numpy())

from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_true, preds)
print(f"Error absoluto medio (MAE): ${mae:.2f}")