import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

# 1. Cargar datos
df = pd.read_csv(r'C:\Users\User win10pro\Documents\herramientas\herramientas-2025\aaa_2\BMW.csv')

# 2. Seleccionar columnas
cat_cols = ['Model', 'Region', 'Color', 'Fuel_Type', 'Transmission']
num_cols = ['Year', 'Engine_Size_L', 'Mileage_KM']
target = 'Price_USD'

# 3. Codificar categóricas
for col in cat_cols:
    df[col] = df[col].astype('category').cat.codes

# 4. Separar X e y
X_cat = df[cat_cols].values
X_num = df[num_cols].values
y = df[target].values.reshape(-1, 1)

# 5. Escalar numéricas
scaler_num = StandardScaler()
scaler_y = StandardScaler()

X_num = scaler_num.fit_transform(X_num)
y = scaler_y.fit_transform(y)

# 6. División
X_cat_train, X_cat_test, X_num_train, X_num_test, y_train, y_test = train_test_split(
    X_cat, X_num, y, test_size=0.2, random_state=42)

# 7. A PyTorch
X_cat_train = torch.tensor(X_cat_train, dtype=torch.long)
X_num_train = torch.tensor(X_num_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)

X_cat_test = torch.tensor(X_cat_test, dtype=torch.long)
X_num_test = torch.tensor(X_num_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

train_dataset = TensorDataset(X_cat_train, X_num_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# 8. Red con embeddings
class BMWModel(nn.Module):
    def __init__(self, emb_dims, n_cont):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(categories, dim) for categories, dim in emb_dims])
        self.n_emb = sum(e.embedding_dim for e in self.embeddings)
        self.n_cont = n_cont

        self.lin1 = nn.Linear(self.n_emb + self.n_cont, 128)
        self.lin2 = nn.Linear(128, 64)
        self.lin3 = nn.Linear(64, 32)
        self.lin4 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()

    def forward(self, x_cat, x_num):
        x = [e(x_cat[:, i]) for i, e in enumerate(self.embeddings)]
        x = torch.cat(x, 1)
        x = torch.cat([x, x_num], 1)
        x = self.relu(self.lin1(x))
        x = self.dropout(x)
        x = self.relu(self.lin2(x))
        x = self.dropout(x)
        x = self.relu(self.lin3(x))
        x = self.lin4(x)
        return x

# 9. Preparar embeddings
emb_dims = [(df[col].nunique(), min(50, (df[col].nunique() + 1) // 2)) for col in cat_cols]
model = BMWModel(emb_dims, len(num_cols))

# 10. Entrenamiento
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 100
for epoch in range(epochs):
    model.train()
    for x_cat, x_num, yb in train_loader:
        pred = model(x_cat, x_num)
        loss = criterion(pred, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# 11. Evaluación
model.eval()
with torch.no_grad():
    preds = model(X_cat_test, X_num_test)
    preds = scaler_y.inverse_transform(preds.numpy())
    y_true = scaler_y.inverse_transform(y_test.numpy())

from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_true, preds)
print(f"Error absoluto medio (MAE) mejorado: ${mae:.2f}")