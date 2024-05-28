import os
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64

def plot_confusion_matrix(conf_matrix, labels, title, filename):
    # Ensure the static directory exists
    static_dir = 'app/static'
    if not os.path.exists(static_dir):
        os.makedirs(static_dir)

    filepath = os.path.join(static_dir, filename)
    plt.figure(figsize=(5, 4))  # Adjust the size here
    # Remove empty columns and rows from the confusion matrix
    conf_matrix = conf_matrix[~(conf_matrix == 0).all(1)]
    conf_matrix = conf_matrix[:, ~(conf_matrix == 0).all(0)]
    if conf_matrix.size == 0:
        print("Confusion matrix is empty, skipping plot.")
        return

    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.savefig(filepath)
    plt.close()

def save_output_csv(df, filename):
    # Ensure the outputs directory exists
    outputs_dir = 'app/outputs'
    if not os.path.exists(outputs_dir):
        os.makedirs(outputs_dir)

    filepath = os.path.join(outputs_dir, filename)
    df.to_csv(filepath, index=False)
    return filepath

def plot_confusion_matrix(conf_matrix, labels, title):
    
    plt.figure(figsize=(5, 4))  # Adjust the size here
    # Remove empty columns and rows from the confusion matrix
    conf_matrix = conf_matrix[~(conf_matrix == 0).all(1)]
    conf_matrix = conf_matrix[:, ~(conf_matrix == 0).all(0)]
    if conf_matrix.size == 0:
        print("Confusion matrix is empty, skipping plot.")
        return None

    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    img = BytesIO()
    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)
    return img
