import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans


# KMeans clustering (MiniBatch)
def kmeansClustering(imgPath, nClusters=64):
    # Read in our image
    im = cv2.imread(imgPath)[:, :, ::-1]

    # Turn the image into an array
    imArray = np.array(im)

    # Make the array 2-dimensional so it can be turned into a DataFrame
    rows, cols, _ = imArray.shape
    imArray2D = imArray.reshape((rows * cols, 3))
    df = pd.DataFrame(imArray2D, columns=["R", "G", "B"])

    # -----------------INFO-----------------
    # All 3 columns are uint8
    # print(df.dtypes)
    # Takes up 9579615 bytes to store this DataFrame
    # print(df.memory_usage(index=True, deep=True).sum())
    # We also can confirm our reshaping didn't mess up the data. The original image was 2159x1479x3 so we'd expect after reshaping to have a 3,193,161x3 DataFrame which is indeed the case
    # print(df.shape)

    # Clustering
    kmeans = MiniBatchKMeans(n_clusters=64, n_init=40)
    kmeans.fit(df)

    # Note, the centers are most likely not integers, so we'll need to approximate them
    centers = np.round(kmeans.cluster_centers_).astype(np.uint8)
    labels = kmeans.labels_
    compressed_img_array = centers[labels]

    # Change the shape back to normal (Uses rows and column values from initial import)
    compressed_img = compressed_img_array.reshape((rows, cols, 3))

    return im, compressed_img


# For viewing clusters in 3-Dimensions (Lags computer even at 250)
def plot_rgb_clusters_3d(df, kmeans, sample_size=250):
    """
    Visualizes clustered RGB data in 3D.

    Parameters:
        df (pd.DataFrame): DataFrame with 'R', 'G', 'B' columns.
        kmeans (KMeans): Fitted KMeans object.
        sample_size (int, optional): If provided, randomly samples this many points per cluster to plot.
    """
    df_viz = df.copy()
    df_viz["cluster"] = kmeans.labels_

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    for i in range(kmeans.n_clusters):
        cluster_points = df_viz[df_viz["cluster"] == i][["R", "G", "B"]]
        if sample_size is not None and len(cluster_points) > sample_size:
            cluster_points = cluster_points.sample(sample_size)
        color = kmeans.cluster_centers_[i] / 255.0  # Normalize to [0, 1]
        ax.scatter(cluster_points["R"], cluster_points["G"], cluster_points["B"],
                   color=color, label=f'Cluster {i}', s=1)

    ax.set_xlabel("R")
    ax.set_ylabel("G")
    ax.set_zlabel("B")
    ax.set_title("3D RGB Color Clusters")
    plt.tight_layout()
    plt.show()


# For viewing the images next to each other
def side_by_side(original, compressed):
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(original)
    ax[0].set_title("Original Image")
    ax[0].axis("off")
    ax[1].imshow(compressed)
    ax[1].set_title("Compressed Image")
    ax[1].axis("off")
    plt.tight_layout()
    plt.show()


images = ["RGB.png", "Starry Night.jpg", "Color Explosions.jpg"]
for i in range(len(images)):
    original, modified = kmeansClustering(images[i])
    side_by_side(original, modified)
