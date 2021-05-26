import matplotlib.pyplot as plt
import torch
from visualisation import plot_level_lines


def train(model, x, y, additional_features, criterion, optimizer, n_epochs, device='cpu'):
    data = torch.FloatTensor(x).to(device)
    labels = torch.FloatTensor(y).to(device)
    dataset = torch.utils.data.TensorDataset(data, labels)
    dl = torch.utils.data.DataLoader(dataset, batch_size=32, pin_memory=True, num_workers=2)
    for epoch in range(n_epochs):
        
        model.train()
        for batch in dl:
            optimizer.zero_grad()
            xs, ys = batch
            pred = model(xs).squeeze()
            loss = criterion(pred, ys)
            loss.backward()
            optimizer.step()
        
        model.eval()
        pred = (model(data).squeeze() >= 0.5).float()
        acc = (pred == labels).float().mean()
            
        plot_level_lines(model, data[:, :2], labels, additional_features, epoch=epoch, acc=acc)
        plt.show()