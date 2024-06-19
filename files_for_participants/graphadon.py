from torch_geometric.nn import GCNConv, GraphConv, GATConv
from torch_geometric.nn import global_mean_pool, global_max_pool
from torch_geometric.nn import Linear
import torch

from tqdm import tqdm

class FirstGNN(torch.nn.Module):
    def __init__(self, in_channels=-1, hidden_channels=64, out_channels=2):
        super(FirstGNN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        
        x = global_mean_pool(x, batch)  

        x = self.lin(x)
        
        return x
    
    

def test_model(model, loader):
    model.eval()
    correct = 0
    for data in loader:
        out = model(data.x, data.edge_index, data.batch)
        pred = out.argmax(dim=1)
        correct += (pred == data.y).sum().item()
    return correct / len(loader.dataset)  

def learn_and_perf(model_to_test, train_loader, test_loader=None, nb_epochs=1000):
    
    optimizer = torch.optim.Adam(model_to_test.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    losses = []
    accs_train = []
    if test_loader is not None:    
        accs_test =  []
    else:
        accs_test = None

    for _ in tqdm(range(nb_epochs)):
        accs_train.append(test_model(model_to_test, train_loader))
        if test_loader is not None:
            accs_test.append(test_model(model_to_test, test_loader))
        
        model_to_test.train()
        cur_loss = 0
        for data in train_loader:
            out = model_to_test.forward(data.x, data.edge_index, batch=data.batch)  
            loss = criterion(out, data.y)  
            loss.backward()
            cur_loss += loss.item()
            optimizer.step() 
            optimizer.zero_grad() 
        losses.append(cur_loss)

    return accs_train, accs_test, losses
    
def generate_pred_for_kaggle(model, loader):
    model.eval()
    preds = []
    for data in loader:
        out = model(data.x, data.edge_index, data.batch)
        pred = out.argmax(dim=1)
        preds.append(pred)
    return torch.cat(preds)

def generate_kaggle_file(preds, output_file="kaggle.csv"):
    with open(output_file, "w") as f:
        f.write("ID, TARGET\n")
        for i, pred in enumerate(preds):
            f.write(f"{i},{pred.item()}\n")
    