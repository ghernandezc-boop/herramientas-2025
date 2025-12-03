import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from torch.utils.data import TensorDataset, DataLoader

# 1. Cargar datos
df = pd.read_csv(r'C:\Users\User win10pro\Documents\herramientas\herramientas-2025\aaa_2\BMW.csv')

# 2. Codificar todo
df_encoded = df.copy()
for col in df_encoded.select_dtypes(include=['object']).columns:
    df_encoded[col] = LabelEncoder().fit_transform(df_encoded[col].astype(str))

# 3. X e y
X = df_encoded.drop(columns=['Sales_Classification', 'Price_USD'])
y = df_encoded['Sales_Classification']

# 4. División
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. A PyTorch
X_train = torch.tensor(X_train.values, dtype=torch.float32)
X_test = torch.tensor(X_test.values, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.long)
y_test = torch.tensor(y_test.values, dtype=torch.long)

train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# 6. Red neuronal
class Classifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 2)  # 2 clases: High / Low
        )

    def forward(self, x):
        return self.net(x)

model = Classifier(X_train.shape[1])
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 7. Entrenamiento
epochs = 100
for epoch in range(epochs):
    model.train()
    for xb, yb in train_loader:
        pred = model(xb)
        loss = criterion(pred, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# 8. Evaluación
model.eval()
with torch.no_grad():
    preds = model(X_test).argmax(dim=1)
    print(classification_report(y_test, preds, target_names=['Low', 'High']))
    # Fila codificada (según LabelEncoder usado)
import torch
import pandas as pd

# Codificación manual rápida (según el fit previo)
row = {
    'Model': 22,        # X6
    'Year': 13,         # 2023
    'Region': 3,        # Europe
    'Color': 9,         # White
    'Fuel_Type': 2,     # Hybrid
    'Transmission': 0,  # Automatic
    'Engine_Size_L': 3.3,
    'Mileage_KM': 54800,
    'Sales_Volume': 9966
}

# A tensor
input_vec = torch.tensor([list(row.values())], dtype=torch.float32)
model.eval()
with torch.no_grad():
    log_probs = model(input_vec)
    pred_class = log_probs.argmax(dim=1).item()
    confidence = torch.softmax(log_probs, dim=1)[0, pred_class].item()

print("Predicción:", "High" if pred_class == 1 else "Low")
print("Confianza:", f"{confidence:.2%}")