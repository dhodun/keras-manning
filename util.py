import matplotlib.pyplot as plt
import numpy as np
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot



def print_curves(history):
    acc = history.history.get('acc')
    val_acc = history.history.get('val_acc')

    loss = history.history.get('loss')
    val_loss = history.history.get('val_loss')
    
    mae = history.history.get('mean_absolute_error')
    val_mae = history.history.get('val_mean_absolute_error')

    epochs = range(1, len(loss) + 1)
    
    if acc is not None:


        plt.plot(epochs, acc, label='Accuracy')
        plt.plot(epochs, val_acc, label='Validation Accuracy')
        plt.legend()
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('accuracy')

        plt.figure()
        print('Max Val Acc: {}'.format(np.max(val_acc)))
        
    if mae is not None:
        
        plt.plot(epochs, mae, label='MAE')
        plt.plot(epochs, val_mae, label='Validation MAE')
        plt.legend()
        plt.title('Training and Validation MAE')
        plt.xlabel('Epochs')
        plt.ylabel('MAE')

        plt.figure()
        print('Min Val MAE: {}'.format(np.min(val_mae)))

    plt.plot(epochs, loss, label='Loss')
    plt.plot(epochs, val_loss, label='Validation Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    print('Min Val Loss: {}'.format(np.min(val_loss)))

    plt.show()
    return

def plot_model_jupyter(model):
    SVG(model_to_dot(model).create(prog='dot', format='svg'))
    
