from numpy import genfromtxt
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

fn = r'C:\Users\DELL I5558\Desktop\Python\ELEC5222\kMeans\NSW-ER01.csv'
my_data = genfromtxt(fn, delimiter=',')

scaler = StandardScaler()
model = KMeans(n_clusters=4)
pipeline = make_pipeline(scaler, model)
scaler.fit(my_data)
my_data_scaled = scaler.transform(my_data)
model.fit(my_data_scaled)
labels = model.predict(my_data_scaled)

model = TSNE(n_components=3, perplexity=5, learning_rate=100, n_iter=1000, random_state=0)
transformed = model.fit_transform(my_data_scaled)
xs = transformed[:, 0]
ys = transformed[:, 1]
zs = transformed[:, 2]
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xs, ys, zs, c=labels, marker='o')
plt.show()
