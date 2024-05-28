import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

def plot_confusion_matrix(conf_matrix, labels, title):
    plt.figure(figsize=(5, 4))  # Adjust the size here

    conf_matrix = conf_matrix[~(conf_matrix == 0).all(1)]
    conf_matrix = conf_matrix[:, ~(conf_matrix == 0).all(0)]
    if conf_matrix.size == 0:
        print("Confusion matrix is empty, skipping plot.")
        return None

    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png')
    plt.close()
    img_buffer.seek(0)
    img_data = img_buffer.getvalue()
    img_base64 = base64.b64encode(img_data).decode()
    return img_base64
