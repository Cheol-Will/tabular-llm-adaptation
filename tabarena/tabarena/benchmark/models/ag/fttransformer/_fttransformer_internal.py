import time
import torch
import logging
import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from torch.utils.data import DataLoader, TensorDataset
from . import fttransformer_reference as ftt_ref

logger = logging.getLogger(__name__)

class FTTransformerImplementation:
    def __init__(self, **config):
        self.config = config
        self.model = None
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='mean')
        self.device = torch.device(config.get('device', 'cpu'))
        self.has_num_cols = False
        if 'n_threads' in config:
            torch.set_num_threads(config['n_threads'])

    def fit(self, X_train, y_train, X_val, y_val, cat_col_names, time_to_fit_in_seconds=None):
        start_time = time.time()
        num_cols = [c for c in X_train.columns if c not in cat_col_names]
        self.has_num_cols = len(num_cols) > 0
        
        if self.has_num_cols:
            xt_num_raw = self.imputer.fit_transform(X_train[num_cols].to_numpy(dtype=np.float32))
            xt_num = torch.as_tensor(self.scaler.fit_transform(xt_num_raw), dtype=torch.float32)
            
            xv_num_raw = self.imputer.transform(X_val[num_cols].to_numpy(dtype=np.float32))
            xv_num = torch.as_tensor(self.scaler.transform(xv_num_raw), dtype=torch.float32)
        else:
            xt_num = torch.empty((len(X_train), 0), dtype=torch.float32)
            xv_num = torch.empty((len(X_val), 0), dtype=torch.float32)

        xt_cat = torch.as_tensor(X_train[cat_col_names].values.astype(np.int64), dtype=torch.long)
        xv_cat = torch.as_tensor(X_val[cat_col_names].values.astype(np.int64), dtype=torch.long)
        
        yt = torch.as_tensor(y_train.values, dtype=torch.float32 if self.config['task'] == 'reg' else torch.long)
        yv = torch.as_tensor(y_val.values, dtype=torch.float32 if self.config['task'] == 'reg' else torch.long)

        self.model = ftt_ref.FTTransformer(
            n_num=len(num_cols), 
            cat_cards=self.config['cards'], 
            n_out=self.config['n_out'],
            n_blocks=self.config.get('n_blocks', 3), 
            d_token=self.config.get('d_token', 64),
            n_heads=self.config.get('n_heads', 4), 
            d_ffn_factor=4/3, 
            dropout=self.config.get('dropout', 0.1)
        ).to(self.device)

        opt = torch.optim.AdamW(self.model.parameters(), lr=self.config.get('lr', 1e-4), weight_decay=1e-5)
        loss_fn = torch.nn.MSELoss() if self.config['task'] == 'reg' else torch.nn.CrossEntropyLoss()
        
        loader = DataLoader(TensorDataset(xt_num, xt_cat, yt), batch_size=self.config.get('batch_size', 256), shuffle=True)
        
        for epoch in range(self.config.get('epochs', 32)):
            if time_to_fit_in_seconds and (time.time() - start_time) > time_to_fit_in_seconds:
                break
                
            self.model.train()
            for bn, bc, by in loader:
                opt.zero_grad()
                out = self.model(bn.to(self.device), bc.to(self.device)).squeeze()
                loss = loss_fn(out, by.to(self.device))
                loss.backward()
                opt.step()
            
            if self.config.get('verbosity', 0) >= 2:
                logger.info(f"Epoch {epoch} complete")

    def predict(self, X, cat_col_names):
        self.model.eval()
        num_cols = [c for c in X.columns if c not in cat_col_names]
        
        if self.has_num_cols:
            xn_raw = self.imputer.transform(X[num_cols].to_numpy(dtype=np.float32))
            xn = torch.as_tensor(self.scaler.transform(xn_raw), dtype=torch.float32).to(self.device)
        else:
            xn = torch.empty((len(X), 0), dtype=torch.float32).to(self.device)
            
        xc = torch.as_tensor(X[cat_col_names].values.astype(np.int64), dtype=torch.long).to(self.device)
        
        with torch.no_grad():
            return self.model(xn, xc).cpu().numpy()