import matplotlib.pyplot as plt
import seaborn as sns
import os

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
