import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(y_test, y_pred, label_encoder, save_path="results/plots/confusion_matrix.png"):
    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Get class names
    class_names = label_encoder.classes_

    # Create figure
    plt.figure(figsize=(8, 6))

    # Create heatmap
    sns.heatmap(cm,
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names,
                cbar_kws={'label': 'Count'},
                square=True)

    # Add labels and title
    plt.title('Confusion Matrix - Credit Grade Classification', fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Predicted Credit Grade', fontsize=12, fontweight='bold')
    plt.ylabel('Actual Credit Grade', fontsize=12, fontweight='bold')

    # Add accuracy text
    accuracy = np.sum(np.diag(cm)) / np.sum(cm)
    plt.figtext(0.02, 0.02, f'Overall Accuracy: {accuracy:.2%}',
                fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8))

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()

    return cm