import numpy as np
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


def tsne(X_scaled, y):
    tsne = TSNE(n_components=3, verbose=1, random_state=42)
    result = tsne.fit_transform(X_scaled)

    scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    result = scaler.fit_transform(result)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(result[:, 0], result[:, 1], result[:, 2], c=y)
    ax.set_proj_type('ortho')
    plt.show()

def pca(X_scaled, y):
    pca = PCA(n_components=10)
    X_pca = pca.fit_transform(X_scaled)

    plt.figure(figsize=(10, 8))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis')
    plt.title('PCA')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.colorbar(label='Is good')
    plt.show()

    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")


if __name__ == '__main__':

    x = np.load('./embeddings.npy').reshape(-1, 1536)
    y = np.load('./label.npy').reshape(-1)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(x)
    pca(X_scaled, y)

