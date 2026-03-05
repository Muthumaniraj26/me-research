import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# --- 1. ARCHITECTURE COMPONENTS ---

class TemporalGate(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(dim, dim), nn.Sigmoid())
    def forward(self, x):
        return x * self.gate(x)

class GrandMasterHybrid(nn.Module):
    def __init__(self, seq_len, num_vars):
        super().__init__()
        self.cnn_local = nn.Conv1d(num_vars, 64, kernel_size=3, padding=1)
        self.cnn_dilated = nn.Conv1d(num_vars, 64, kernel_size=3, padding=2, dilation=2)
        self.bn1 = nn.BatchNorm1d(128)
        
        self.inv_emb = nn.Linear(seq_len, 128)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=128, nhead=8, dim_feedforward=512, dropout=0.05, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        self.gate = TemporalGate(128)
        self.gru = nn.GRU(128, 64, batch_first=True, bidirectional=True)
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        self.decoder = nn.Sequential(
            nn.Linear(128, 256), nn.GELU(), nn.Dropout(0.1), nn.Linear(256, 1)
        )

    def forward(self, x):
        x_in = x.transpose(1, 2)
        x_cnn = torch.cat([self.cnn_local(x_in), self.cnn_dilated(x_in)], dim=1)
        x_cnn = self.bn1(torch.relu(x_cnn))
        x_inv = self.inv_emb(x_cnn)
        x_trans = self.transformer(x_inv)
        x_gated = self.gate(x_trans)
        x_liq, _ = self.gru(x_gated)
        x_pool = self.pool(x_liq.transpose(1, 2)).squeeze(-1)
        return self.decoder(x_pool)

# --- 2. EVALUATION FUNCTION ---

def get_metrics(actual, predicted):
    # Flatten for metric calculation
    actual, predicted = actual.flatten(), predicted.flatten()
    r2 = r2_score(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mse)
    # Matching your Precision metric: 1 - (MAE / Mean)
    precision = 1 - (mae / (np.mean(actual) + 1e-9))
    return r2, rmse, mae, mse, precision

# --- 3. EXECUTION ---

def run_research_eval():
    samples, seq_len, vars = 5000, 24, 5
    batch_size = 64
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data Simulation (Using your solar pattern logic)
    t = np.linspace(0, 200, samples)
    y_raw = np.sin(t) + 0.3 * np.cos(t/2) + np.random.normal(0, 0.01, samples)
    X_raw = np.tile(y_raw.reshape(-1, 1), (1, vars)) + np.random.normal(0, 0.005, (samples, vars))

    scaler_x, scaler_y = RobustScaler(), RobustScaler()
    X_scaled = scaler_x.fit_transform(X_raw)
    y_scaled = scaler_y.fit_transform(y_raw.reshape(-1, 1))

    X_win, y_win = [], []
    for i in range(len(X_scaled) - seq_len):
        X_win.append(X_scaled[i:i+seq_len])
        y_win.append(y_scaled[i+seq_len])

    X_t, y_t = torch.FloatTensor(np.array(X_win)), torch.FloatTensor(np.array(y_win))
    split = int(0.9 * len(X_t))
    
    train_loader = DataLoader(TensorDataset(X_t[:split], y_t[:split]), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_t[split:], y_t[split:]), batch_size=batch_size)

    model = GrandMasterHybrid(seq_len, vars).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.HuberLoss() # More stable for high R2 than MSE

    print(f"Training Final Hybrid on {device}...")
    for epoch in range(50):
        model.train()
        for b_x, b_y in train_loader:
            b_x, b_y = b_x.to(device), b_y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(b_x), b_y)
            loss.backward()
            optimizer.step()

    # --- FINAL EVALUATION ---
    model.eval()
    def get_all_preds(loader):
        preds, actuals = [], []
        with torch.no_grad():
            for b_x, b_y in loader:
                b_x = b_x.to(device)
                out = model(b_x).cpu().numpy()
                act = b_y.numpy()
                preds.append(scaler_y.inverse_transform(out))
                actuals.append(scaler_y.inverse_transform(act))
        return np.vstack(actuals), np.vstack(preds)

    train_act, train_pred = get_all_preds(train_loader)
    test_act, test_pred = get_all_preds(test_loader)

    tr_r2, tr_rmse, tr_mae, tr_mse, tr_prec = get_metrics(train_act, train_pred)
    te_r2, te_rmse, te_mae, te_mse, te_prec = get_metrics(test_act, test_pred)

    # --- PRINT THE TABLE ---
    results = {
        "Model": ["Grand-Master Hybrid"],
        "Train_Acc(R2)": [f"{tr_r2:.8f}"],
        "Test_Acc(R2)": [f"{te_r2:.8f}"],
        "RMSE": [f"{te_rmse:.8f}"],
        "MAE": [f"{te_mae:.8f}"],
        "MSE": [f"{te_mse:.8f}"],
        "Precision*": [f"{te_prec:.8f}"]
    }
    
    print("\n" + "="*80)
    print(pd.DataFrame(results).to_string(index=False))
    print("="*80)

if __name__ == "__main__":
    run_research_eval()