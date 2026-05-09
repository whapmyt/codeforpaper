
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

def create_clusters(X, k=5):
    pca = PCA(n_components=10)
    X_reduced = pca.fit_transform(X)
    kmeans = KMeans(n_clusters=k, random_state=42)
    clusters = kmeans.fit_predict(X_reduced)
    return clusters
