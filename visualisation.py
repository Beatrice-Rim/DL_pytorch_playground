import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
from scipy.special import expit
import torch


def plot_level_lines(model, data, labels, additional_features, size=100, make_new_figure=True, scatter=True, epoch=None, acc=None):
    clear_output(wait=True)
    if make_new_figure:
        plt.figure(figsize=(9, 6))
    x_min = data[:, 0].min() - 0.1
    x_max = data[:, 0].max() + 0.1
    y_min = data[:, 1].min() - 0.1
    y_max = data[:, 1].max() + 0.1
    all_x = np.linspace(x_min, x_max, size)
    all_y = np.linspace(y_min, y_max, size)
    XX, YY = np.meshgrid(all_x, all_y)
    test_data = np.c_[XX.ravel(), YY.ravel()]
    
    add = additional_features
    
    if add == 'original':
        test_data = test_data
    if add == 'squares':
        test_data = np.hstack([test_data, test_data**2, test_data[:, [0]]*test_data[:, [1]]])
    if add == 'sin':
        test_data = np.hstack([test_data, np.sin(test_data)])
    if add == 'both':
        test_data = np.hstack([test_data, test_data**2, test_data[:, [0]]*test_data[:, [1]], np.sin(test_data)])

    model.eval()
    with torch.no_grad():
        predictions = model(torch.FloatTensor(test_data)).cpu().numpy().reshape((size, size))  # these are raw scores
    
    predictions = expit(predictions)
    plt.contourf(all_x, all_y, predictions)
    plt.colorbar()
    
    plt.contour(XX, YY, predictions, levels=[0.5], linewidths=2, colors='k')

    if scatter:
        plt.scatter(data[:, 0], data[:, 1], alpha=0.7)
    plt.title(f"Epoch: {epoch}, Accuracy: {acc}")
    axes = plt.gca()
    axes.set_xlim([x_min,x_max])
    axes.set_ylim([y_min,y_max])