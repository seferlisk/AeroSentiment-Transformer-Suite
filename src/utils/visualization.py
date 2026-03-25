import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


class Visualizer:
    @staticmethod
    def plot_learning_curves(history):
        """
        Plots training & validation loss and accuracy.
        """
        epochs = range(1, len(history['train_loss']) + 1)

        #  Initialize figure via subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Loss Curve
        ax1.plot(epochs, history['train_loss'], 'b-o', label='Training Loss')
        ax1.plot(epochs, history['val_loss'], 'r-o', label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.legend()

        # Accuracy Curve
        ax2.plot(epochs, history['train_acc'], 'b-o', label='Training Acc')
        ax2.plot(epochs, history['val_acc'], 'r-o', label='Validation Acc')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Accuracy')
        ax2.legend()

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, classes):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=classes, yticklabels=classes)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()