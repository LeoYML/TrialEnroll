import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import os
os.environ["OMP_NUM_THREADS"] = "16"
os.environ["MKL_NUM_THREADS"] = "16"
torch.set_num_threads(16)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")


from stack_features_dcn import load_data

# Load your data
x_train_inc_emb, x_train_inc_mask, x_train_excl_emb, x_train_excl_mask, x_test_inc_emb,\
x_test_inc_mask, x_test_excl_emb, x_test_excl_mask, x_train_dense, x_test_dense, x_train_short_onehot,\
x_test_short_onehot, x_train_sparse_country, x_test_sparse_country, x_train_sparse_state,\
x_test_sparse_state, x_train_sparse_city, x_test_sparse_city, y_train, y_test = load_data()



# print(f"x_train_inc_emb: {x_train_inc_emb.shape}, x_train_inc_mask: {x_train_inc_mask.shape}, x_train_excl_emb: {x_train_excl_emb.shape}, x_train_excl_mask: {x_train_excl_mask.shape}, x_test_inc_emb: {x_test_inc_emb.shape}, x_test_inc_mask: {x_test_inc_mask.shape}, x_test_excl_emb: {x_test_excl_emb.shape}, x_test_excl_mask: {x_test_excl_mask.shape}, x_train_dense: {x_train_dense.shape}, x_test_dense: {x_test_dense.shape}, x_train_short_onehot: {x_train_short_onehot.shape}, x_test_short_onehot: {x_test_short_onehot.shape}, x_train_sparse_country: {x_train_sparse_country.shape}, x_test_sparse_country: {x_test_sparse_country.shape}, x_train_sparse_state: {x_train_sparse_state.shape}, x_test_sparse_state: {x_test_sparse_state.shape}, x_train_sparse_city: {x_train_sparse_city.shape}, x_test_sparse_city: {x_test_sparse_city.shape}, y_train: {y_train.shape}, y_test: {y_test.shape}")
# breakpoint()


def prepare_indices(data):
#    return pad_sequence([torch.tensor((row == 1).nonzero(as_tuple=True)[0].tolist(), dtype=torch.long) for row in data], batch_first=True, padding_value=0)
    return data
x_train_sparse_country_indice = prepare_indices(x_train_sparse_country)
x_test_sparse_country_indice = prepare_indices(x_test_sparse_country)
x_train_sparse_state_indice = prepare_indices(x_train_sparse_state)
x_test_sparse_state_indice = prepare_indices(x_test_sparse_state)
x_train_sparse_city_indice = prepare_indices(x_train_sparse_city)
x_test_sparse_city_indice = prepare_indices(x_test_sparse_city)




y_train = torch.tensor(y_train, dtype=torch.int)
y_test = torch.tensor(y_test, dtype=torch.int)

from imblearn.over_sampling import RandomOverSampler

# Combine the 2D training data
x_train_combined_2d = torch.cat([
    x_train_dense.float(),
    x_train_short_onehot.float(),
    x_train_sparse_country.float(),
    x_train_sparse_state.float(),
    x_train_sparse_city.float()
], dim=1)

x_test_combined_2d = torch.cat([
    x_test_dense.float(),
    x_test_short_onehot.float(),
    x_test_sparse_country.float(),
    x_test_sparse_state.float(),
    x_test_sparse_city.float()
], dim=1)

# Apply RandomOverSampler to 2D combined data
ros = RandomOverSampler(random_state=0)
x_train_combined_2d, y_train = ros.fit_resample(x_train_combined_2d, y_train)
resample_indices_train = ros.sample_indices_

ros = RandomOverSampler(random_state=0)
x_test_combined_2d, y_test = ros.fit_resample(x_test_combined_2d, y_test)
resample_indices_test = ros.sample_indices_

# Resample the 3D data using the same indices
x_train_inc_emb = x_train_inc_emb[resample_indices_train]
x_train_inc_mask = x_train_inc_mask[resample_indices_train]
x_train_excl_emb = x_train_excl_emb[resample_indices_train]
x_train_excl_mask = x_train_excl_mask[resample_indices_train]

x_test_inc_emb = x_test_inc_emb[resample_indices_test]
x_test_inc_mask = x_test_inc_mask[resample_indices_test]
x_test_excl_emb = x_test_excl_emb[resample_indices_test]
x_test_excl_mask = x_test_excl_mask[resample_indices_test]

# Define feature lengths for 2D data
num_dense_features = x_train_dense.shape[1]
num_short_onehot_features = x_train_short_onehot.shape[1]
num_sparse_country_features = x_train_sparse_country.shape[1]
num_sparse_state_features = x_train_sparse_state.shape[1]
num_sparse_city_features = x_train_sparse_city.shape[1]

# Split combined 2D data back into original components
x_train_dense = x_train_combined_2d[:, :num_dense_features]
x_test_dense = x_test_combined_2d[:, :num_dense_features]

start = num_dense_features
x_train_short_onehot = x_train_combined_2d[:, start:start + num_short_onehot_features]
x_test_short_onehot = x_test_combined_2d[:, start:start + num_short_onehot_features]

start += num_short_onehot_features
x_train_sparse_country = x_train_combined_2d[:, start:start + num_sparse_country_features]
x_test_sparse_country = x_test_combined_2d[:, start:start + num_sparse_country_features]

start += num_sparse_country_features
x_train_sparse_state = x_train_combined_2d[:, start:start + num_sparse_state_features]
x_test_sparse_state = x_test_combined_2d[:, start:start + num_sparse_state_features]

start += num_sparse_state_features
x_train_sparse_city = x_train_combined_2d[:, start:start + num_sparse_city_features]
x_test_sparse_city = x_test_combined_2d[:, start:start + num_sparse_city_features]

# Convert y_train and y_test to tensors
y_train = torch.tensor(y_train, dtype=torch.int).to(device)
y_test = torch.tensor(y_test, dtype=torch.int).to(device)

# Move the 2D data to device
x_train_dense = torch.tensor(x_train_dense, dtype=torch.float32).to(device)
x_train_short_onehot = torch.tensor(x_train_short_onehot, dtype=torch.float32).to(device)
x_train_sparse_country = torch.tensor(x_train_sparse_country, dtype=torch.int).to(device)
x_train_sparse_state = torch.tensor(x_train_sparse_state, dtype=torch.int).to(device)
x_train_sparse_city = torch.tensor(x_train_sparse_city, dtype=torch.int).to(device)

x_test_dense = torch.tensor(x_test_dense, dtype=torch.float32).to(device)
x_test_short_onehot = torch.tensor(x_test_short_onehot, dtype=torch.float32).to(device)
x_test_sparse_country = torch.tensor(x_test_sparse_country, dtype=torch.int).to(device)
x_test_sparse_state = torch.tensor(x_test_sparse_state, dtype=torch.int).to(device)
x_test_sparse_city = torch.tensor(x_test_sparse_city, dtype=torch.int).to(device)

# Move the 3D data to device
x_train_inc_emb = x_train_inc_emb.to(device)
x_train_inc_mask = x_train_inc_mask.to(device)
x_train_excl_emb = x_train_excl_emb.to(device)
x_train_excl_mask = x_train_excl_mask.to(device)

x_test_inc_emb = x_test_inc_emb.to(device)
x_test_inc_mask = x_test_inc_mask.to(device)
x_test_excl_emb = x_test_excl_emb.to(device)
x_test_excl_mask = x_test_excl_mask.to(device)




# print(f"x_train_dense: {x_train_dense.shape}, x_train_short_onehot: {x_train_short_onehot.shape}, x_train_sparse_country: {x_train_sparse_country.shape}, x_train_sparse_state: {x_train_sparse_state.shape}, x_train_sparse_city: {x_train_sparse_city.shape}, x_train_inc_emb: {x_train_inc_emb.shape}, x_train_inc_mask: {x_train_inc_mask.shape}, x_train_excl_emb: {x_train_excl_emb.shape}, x_train_excl_mask: {x_train_excl_mask.shape}, y_train: {y_train.shape}")
# print(f"x_test_dense: {x_test_dense.shape}, x_test_short_onehot: {x_test_short_onehot.shape}, x_test_sparse_country: {x_test_sparse_country.shape}, x_test_sparse_state: {x_test_sparse_state.shape}, x_test_sparse_city: {x_test_sparse_city.shape}, x_test_inc_emb: {x_test_inc_emb.shape}, x_test_inc_mask: {x_test_inc_mask.shape}, x_test_excl_emb: {x_test_excl_emb.shape}, x_test_excl_mask: {x_test_excl_mask.shape}, y_test: {y_test.shape}")

# breakpoint()





import torch
import torch.nn as nn

class CriteriaModel(nn.Module):
    def __init__(self):
        super(CriteriaModel, self).__init__()
        
        # dims
        self.extra_features_dim = 57  # short_onehot
        self.embedding_dim = 768  # BERT embedding size
        self.dense_embedding_size = 4 * self.embedding_dim  # 4 BERT embeddings
        self.sparse_embedding_size = 147 + 3799 + 27251  # Country, State, City
        self.total_input_size = self.embedding_dim + self.dense_embedding_size + self.extra_features_dim + self.sparse_embedding_size # 768*5+57+(50+200+300)=5120
        # self.total_input_size = self.embedding_dim + self.dense_embedding_size + self.extra_features_dim (if sparse features aren't included)
        
        # 'CLS' token
        self.cls_token = nn.Parameter(torch.rand(1, 1, self.embedding_dim))
        
        # Transformer Encoder Layer
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embedding_dim, nhead=2, dropout=0.2, batch_first=True, dim_feedforward=2*self.embedding_dim)
        layer_norm = nn.LayerNorm(self.embedding_dim)
        self.transformer_encoder = nn.TransformerEncoder(
            self.transformer_encoder_layer, num_layers=1, norm=layer_norm)

        # MLP with additional layers
        self.fc1 = nn.Linear(self.total_input_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.relu = nn.ReLU()

    def forward(self, incl_emb, incl_mask, excl_emb, excl_mask, dense_embeddings, short_onehot, sparse_embeddings):
        
        # Transformer Encoder
        criteria_emb = torch.cat((incl_emb, excl_emb), dim=1)  # batch*64*768
        criteria_emb = torch.cat((self.cls_token.expand(criteria_emb.shape[0], -1, -1), criteria_emb), dim=1)  # batch*65*768
        criteria_mask = torch.cat((incl_mask, excl_mask), dim=1)  # batch*64
        criteria_mask = torch.cat((torch.ones(criteria_emb.shape[0], 1, dtype=torch.bool, device=criteria_emb.device), criteria_mask), dim=1)  # batch*65
        criteria_mask = criteria_mask == 0 
        
        criteria_encoded = self.transformer_encoder(criteria_emb, src_key_padding_mask=criteria_mask)  # batch*65*768
        
        # 'CLS' token
        pooled_emb = criteria_encoded[:, 0, :]  # batch*768  # pool_emb is nan without "criteria_mask = criteria_mask == 0"
        
        #print(f"pooled_emb: {pooled_emb}")
        
        # Concatenate features
        concatenated = torch.cat((pooled_emb, dense_embeddings, short_onehot, sparse_embeddings), dim=1)


        # MLP Forward Pass
        output = self.relu(self.fc1(concatenated))
        output = self.relu(self.fc2(output))
        output = self.fc3(output)


        return output




class CrossNet(nn.Module):
    def __init__(self, in_features: int, num_layers: int) -> None:
        super(CrossNet, self).__init__()
        self._num_layers = num_layers
        self.kernels = nn.ParameterList(
            [nn.Parameter(torch.empty(in_features, in_features)) for _ in range(self._num_layers)]
        )
        self.bias = nn.ParameterList(
            [nn.Parameter(torch.empty(in_features, 1)) for _ in range(self._num_layers)]
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x_0 = input.unsqueeze(2)  # (B, N, 1)
        x_l = x_0

        for layer in range(self._num_layers):
            xl_w = torch.matmul(self.kernels[layer], x_l)  # (B, N, 1)
            x_l = x_0 * (xl_w + self.bias[layer]) + x_l  # (B, N, 1)

        return torch.squeeze(x_l, dim=2)


class DCN(nn.Module):
    def __init__(self, sparse_input_dims, cross_layers):
        self.pooled_embedding_dim = 768
        self.extra_features_dim = 57 # short_onehot
        self.dense_embedding_dims = 4 * 768 # 4 BERT embeddings
        
        super(DCN, self).__init__()
        #total_embedding_dim = self.pooled_embedding_dim + self.dense_embedding_dims + self.extra_features_dim + sum(sparse_embedding_dims) 
        #total_embedding_dim = self.pooled_embedding_dim + self.dense_embedding_dims + self.extra_features_dim
        self.cross_network = CrossNet(57, cross_layers) #只用extra_features+country onehot
        self.criteria_model = CriteriaModel()
        
        #self.final_linear = nn.Linear(total_embedding_dim + total_embedding_dim//2, 1)
        self.final_linear = nn.Linear(128+57, 1)
        
        # Initialize initial weights
        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'embedding' in name:
                nn.init.uniform_(param, -0.1, 0.1)
            elif 'weight' in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, incl_emb, incl_mask, excl_emb, excl_mask, dense_embeddings, short_onehot, sparse_embeds):
        

    
        sparse_embeddings = torch.cat(sparse_embeds, dim=1)
        cross_input = short_onehot
        cross_out = self.cross_network(cross_input)
        # sigmoid on cross out
        cross_out = torch.sigmoid(cross_out)
        
        
        criteria_out = self.criteria_model(incl_emb, incl_mask, excl_emb, excl_mask, dense_embeddings, short_onehot, sparse_embeddings)
        #criteria_out = self.criteria_model(incl_emb, incl_mask, excl_emb, excl_mask, dense_embeddings, short_onehot)
        
        
        stack_output = torch.cat([cross_out, criteria_out], dim=1)
        #stack_output = criteria_out
        output = self.final_linear(stack_output)
        # print(f"cross_out: {cross_out}, criteria_out: {criteria_out}")
        # print(f"output: {output}")
        # breakpoint()
        return output



from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, x_dense, x_short_onehot, x_sparse_country, x_sparse_state, x_sparse_city, 
                 x_inc_emb, x_inc_mask, x_excl_emb, x_excl_mask, y):
        self.x_dense = x_dense
        self.x_short_onehot = x_short_onehot
        self.x_sparse_country = x_sparse_country
        self.x_sparse_state = x_sparse_state
        self.x_sparse_city = x_sparse_city
        self.x_inc_emb = x_inc_emb
        self.x_inc_mask = x_inc_mask
        self.x_excl_emb = x_excl_emb
        self.x_excl_mask = x_excl_mask
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        sample = {
            'dense': self.x_dense[idx],
            'short_onehot': self.x_short_onehot[idx],
            'sparse_country': self.x_sparse_country[idx],
            'sparse_state': self.x_sparse_state[idx],
            'sparse_city': self.x_sparse_city[idx],
            'inc_emb': self.x_inc_emb[idx],
            'inc_mask': self.x_inc_mask[idx],
            'excl_emb': self.x_excl_emb[idx],
            'excl_mask': self.x_excl_mask[idx],
            'label': self.y[idx]
        }
        return sample



# Hyperparameters
sparse_feature_dims = [147, 3799, 27251]
num_epochs = 15


def train(params):
    print(f"params: {params}")

    batch_size = params['batch_size']
    cross_layers = params['cross_layers']
    learning_rate = params['lr']
    weight_decay = params['weight_decay']

    # Initialize model, loss function, and optimizer
    model = DCN(sparse_feature_dims, cross_layers).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Create CustomDataset and DataLoader for efficient batch loading
    train_dataset = CustomDataset(
        x_train_dense, x_train_short_onehot, x_train_sparse_country, x_train_sparse_state, x_train_sparse_city,
        x_train_inc_emb, x_train_inc_mask, x_train_excl_emb, x_train_excl_mask, y_train
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = CustomDataset(
        x_test_dense, x_test_short_onehot, x_test_sparse_country, x_test_sparse_state, x_test_sparse_city,
        x_test_inc_emb, x_test_inc_mask, x_test_excl_emb, x_test_excl_mask, y_test
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Training loop
    best_auc = 0
        

    patience = 2
    bad_epoch = 0

    for epoch in tqdm(range(num_epochs)):
        model.train()
        for batch in train_loader:
            batch_x_dense = batch['dense'].to(device)
            batch_x_short_onehot = batch['short_onehot'].to(device)
            batch_x_sparse_country = batch['sparse_country'].to(device)
            batch_x_sparse_state = batch['sparse_state'].to(device)
            batch_x_sparse_city = batch['sparse_city'].to(device)
            batch_x_inc_emb = batch['inc_emb'].to(device)
            batch_x_inc_mask = batch['inc_mask'].to(device)
            batch_x_excl_emb = batch['excl_emb'].to(device)
            batch_x_excl_mask = batch['excl_mask'].to(device)
            batch_y = batch['label'].to(device)

            optimizer.zero_grad()
            outputs = model(batch_x_inc_emb, batch_x_inc_mask, batch_x_excl_emb, batch_x_excl_mask,
                            batch_x_dense, batch_x_short_onehot, 
                            [batch_x_sparse_country, batch_x_sparse_state, batch_x_sparse_city])
            
            # print(f"outputs: {outputs}")
            # breakpoint()
            #print(f'Nans: {torch.isnan(outputs).sum()}/64')
            
            
            loss = criterion(outputs, batch_y.unsqueeze(1).float())
            loss.backward()
            optimizer.step()


        model.eval()
        
        with torch.no_grad():
            all_y_true = []
            all_y_pred_probs = []
            for batch in test_loader:
                batch_x_dense = batch['dense'].to(device)
                batch_x_short_onehot = batch['short_onehot'].to(device)
                batch_x_sparse_country = batch['sparse_country'].to(device)
                batch_x_sparse_state = batch['sparse_state'].to(device)
                batch_x_sparse_city = batch['sparse_city'].to(device)
                batch_x_inc_emb = batch['inc_emb'].to(device)
                batch_x_inc_mask = batch['inc_mask'].to(device)
                batch_x_excl_emb = batch['excl_emb'].to(device)
                batch_x_excl_mask = batch['excl_mask'].to(device)
                batch_y = batch['label'].to(device)

                outputs = model(batch_x_inc_emb, batch_x_inc_mask, batch_x_excl_emb, batch_x_excl_mask,
                                batch_x_dense, batch_x_short_onehot, 
                                [batch_x_sparse_country, batch_x_sparse_state, batch_x_sparse_city])
                
                # print(f"outputs: {outputs}")
                # breakpoint()
                # print(f"batch_x_inc_emb: {batch_x_inc_emb}, batch_x_inc_mask: {batch_x_inc_mask}, batch_x_excl_emb: {batch_x_excl_emb}, batch_x_excl_mask: {batch_x_excl_mask}, batch_x_dense: {batch_x_dense}, batch_x_short_onehot: {batch_x_short_onehot}")
                # print(f"outputs: {outputs}")
                #print(f'Nans: {torch.isnan(outputs).sum()}/64')
                #outputs = torch.nan_to_num(outputs, nan=0.0)
                

                
                y_pred_probs = outputs.cpu().numpy().flatten()
                all_y_pred_probs.extend(y_pred_probs)
                all_y_true.extend(batch_y.cpu().numpy().flatten())

            try:
                y_pred_probs = np.array(all_y_pred_probs)
                y_true = np.array(all_y_true)
                y_pred = (y_pred_probs > 0.5).astype(int)
                roc_auc = roc_auc_score(y_true, y_pred_probs)
                precision, recall, _ = precision_recall_curve(y_true, y_pred_probs)
                pr_auc = auc(recall, precision)

                    
    
                # Early Stop
                if pr_auc > best_auc:
                    best_auc = pr_auc

                    bad_epoch = 0
                    print(f'Best PR AUC: {best_auc:.4f}')
                    torch.save(model.state_dict(), f'saves/model.pth')
                    # if pr_auc > 0.7:
                    #     torch.save(model.state_dict(), f'saves/dcn_hatten_only_sampled_prauc={pr_auc}.pth')
                else:
                    bad_epoch += 1

                    if bad_epoch >= patience:
                        print(f'Early stopping at epoch {epoch}')
                        
                        break
                        
                        
            except:
                
                print("Error")
                
                return 0
            
    model.load_state_dict(torch.load('saves/model.pth'))
    model.eval()
    with torch.no_grad():
        all_y_true = []
        all_y_pred_probs = []
        for batch in test_loader:
            batch_x_dense = batch['dense'].to(device)
            batch_x_short_onehot = batch['short_onehot'].to(device)
            batch_x_sparse_country = batch['sparse_country'].to(device)
            batch_x_sparse_state = batch['sparse_state'].to(device)
            batch_x_sparse_city = batch['sparse_city'].to(device)
            batch_x_inc_emb = batch['inc_emb'].to(device)
            batch_x_inc_mask = batch['inc_mask'].to(device)
            batch_x_excl_emb = batch['excl_emb'].to(device)
            batch_x_excl_mask = batch['excl_mask'].to(device)
            batch_y = batch['label'].to(device)

            outputs = model(batch_x_inc_emb, batch_x_inc_mask, batch_x_excl_emb, batch_x_excl_mask,
                            batch_x_dense, batch_x_short_onehot, 
                            [batch_x_sparse_country, batch_x_sparse_state, batch_x_sparse_city])
            
            
            
            y_pred_probs = outputs.cpu().numpy().flatten()
            all_y_pred_probs.extend(y_pred_probs)
            all_y_true.extend(batch_y.cpu().numpy().flatten())

        y_pred_probs = np.array(all_y_pred_probs)
        y_true = np.array(all_y_true)
        y_pred = (y_pred_probs > 0.5).astype(int)
        precision, recall, _ = precision_recall_curve(y_true, y_pred_probs)
        pr_auc = auc(recall, precision)
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        roc_auc = roc_auc_score(y_true, y_pred_probs)
        cm = confusion_matrix(y_true, y_pred)
        
        print(f'Accuracy: {accuracy}\nPrecision: {precision}\nRecall: {recall}\nF1: {f1}\nPR AUC: {pr_auc}\nROC AUC: {roc_auc}\nConfusion Matrix:\n{cm}')

        recall = np.mean(recall)
        
    return accuracy, roc_auc, pr_auc, f1, precision, recall





if __name__ == "__main__":
    

    
    
    params = {'batch_size': 64, 'cross_layers': 1, 'lr': 2.0826702849727576e-05, 'weight_decay': 1e-05}

    accuracy, roc_auc, pr_auc, f1, precision, recall = train(params)
    
        

