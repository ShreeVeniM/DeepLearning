import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix

def save_plot(fig, filename):
    try:
        os.makedirs('charts', exist_ok=True)
        fig.savefig(f'charts/{filename}', dpi=300)
        plt.close(fig)
    except Exception as e:
        print(f"Error in save_plot: {e}")

def plot_loss_curves(history, filename="training_curves.png"):
    try:
        fig, ax = plt.subplots()
        pd.DataFrame(history.history).plot(ax=ax)
        ax.set_title("Model Training Curves")
        save_plot(fig, filename)
    except Exception as e:
        print(f"Error in plot_loss_curves: {e}")

def plot_lr_vs_loss(history, filename="lr_vs_loss.png"):
    try:
        lrs = 1e-5 * (10 ** (np.arange(100)/20))
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.semilogx(lrs, history.history["loss"])
        ax.set_xlabel("Learning Rate")
        ax.set_ylabel("Loss")
        ax.set_title("Learning rate vs. loss")
        save_plot(fig, filename)
    except Exception as e:
        print(f"Error in plot_lr_vs_loss: {e}")

def evaluate_predictions(model, x_test, y_test, filename="confusion_matrix.png"):
    try:
        y_preds = model.predict(x_test)
        y_preds = tf.round(y_preds)
        acc = accuracy_score(y_test, y_preds)
        print(f"Accuracy: {acc}")

        cm = confusion_matrix(y_test, y_preds)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title('Confusion Matrix')
        save_plot(fig, filename)
    except Exception as e:
        print(f"Error in evaluate_predictions: {e}")
