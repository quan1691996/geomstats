import os
import sys
import warnings

import geomstats.backend as gs
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.hypersphere import Hypersphere
import geomstats.datasets.utils as data_utils
from geomstats.geometry.special_orthogonal import SpecialOrthogonal
from geomstats.datasets.prepare_graph_data import HyperbolicEmbedding
import geomstats.geometry.spd_matrices as spd

import matplotlib
import matplotlib.colors as colors
import matplotlib.image as mpimg
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


import geomstats.visualization as visualization

sys.path.append(os.path.dirname(os.getcwd()))
warnings.filterwarnings("ignore")

gs.random.seed(2020)

visualization.tutorial_matplotlib()

# From data on linear spaces to data on manifolds

# Euclidian space
dim = 2
n_samples = 2

euclidean = Euclidean(dim=dim)
points_in_linear_space = euclidean.random_point(n_samples=n_samples)
print("Points in linear space:\n", points_in_linear_space)

linear_mean = gs.sum(points_in_linear_space, axis=0) / n_samples
print("Mean of points:\n", linear_mean)

fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111)

ax.scatter(points_in_linear_space[:, 0], points_in_linear_space[:, 1], label="Points")
ax.plot(points_in_linear_space[:, 0], points_in_linear_space[:, 1], linestyle="dashed")

ax.scatter(
    gs.to_numpy(linear_mean[0]),
    gs.to_numpy(linear_mean[1]),
    label="Mean",
    s=80,
    alpha=0.5,
)

ax.set_title("Mean of points in a linear space")
ax.legend()

# Hypersphere space
sphere = Hypersphere(dim=dim)
points_in_manifold = sphere.random_uniform(n_samples=n_samples)
print("Points in manifold:\n", points_in_manifold)

linear_mean = gs.sum(points_in_manifold, axis=0) / n_samples
print("Mean of points:\n", linear_mean)

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection="3d")

visualization.plot(points_in_manifold, ax=ax, space="S2", label="Point", s=80)

ax.plot(
    points_in_manifold[:, 0],
    points_in_manifold[:, 1],
    points_in_manifold[:, 2],
    linestyle="dashed",
    alpha=0.5,
)

ax.scatter(
    linear_mean[0], linear_mean[1], linear_mean[2], label="Mean", s=80, alpha=0.5
)

ax.set_title("Mean of points on a manifold")
ax.legend()

# Examples of data on manifolds
data, names = data_utils.load_cities()
print(names[:5])
print(data[:5])

gs.all(sphere.belongs(data))

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection="3d")

visualization.plot(data[15:20], ax=ax, space="S2", label=names[15:20], s=80, alpha=0.5)

ax.set_title("Cities on the earth.")

# Pose of objects in pictures: data on the Lie group of 3D rotations
data, img_paths = data_utils.load_poses()

so3 = SpecialOrthogonal(n=3, point_type="vector")

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection="3d")

visualization.plot(data[:2], ax=ax, space="SO3_GROUP")

ax.set_title("3D orientations of the beds.");

# Social networks: data on the hyperbolic space
karate_graph = data_utils.load_karate_graph()

hyperbolic_embedding = HyperbolicEmbedding(max_epochs=20)
embeddings = hyperbolic_embedding.embed(karate_graph)

disk = visualization.PoincareDisk()
fig, ax = plt.subplots(figsize=(8, 8))
disk.set_ax(ax)
disk.draw(ax=ax)
ax.scatter(embeddings[:, 0], embeddings[:, 1])

# Brain connectomes: data on the manifold of Symmetric Positive Definite (SPD) matrices
data, patient_ids, labels = data_utils.load_connectomes()

labels_str = ["Healthy", "Schizophrenic"]

fig = plt.figure(figsize=(8, 4))

ax = fig.add_subplot(121)
imgplot = ax.imshow(data[0])
ax.set_title(labels_str[labels[0]])

ax = fig.add_subplot(122)
imgplot = ax.imshow(data[1])
ax.set_title(labels_str[labels[1]])

manifold = spd.SPDMatrices(28)
gs.all(manifold.belongs(data))

# Monkey’s optical nerve heads: Data as landmarks in 3D
nerves, labels, monkeys = data_utils.load_optical_nerves()
print(nerves.shape)
print(labels)
print(monkeys)

two_nerves = nerves[monkeys == 0]
print(two_nerves.shape)

two_labels = labels[monkeys == 0]
print(two_labels)

label_to_str = {0: "Normal nerve", 1: "Glaucoma nerve"}
label_to_color = {
    0: (102 / 255, 178 / 255, 255 / 255, 1.0),
    1: (255 / 255, 178 / 255, 102 / 255, 1.0),
}

fig = plt.figure()
ax = Axes3D(fig)
ax.set_xlim((2000, 4000))
ax.set_ylim((1000, 5000))
ax.set_zlim((-600, 200))

for nerve, label in zip(two_nerves, two_labels):
    x = nerve[:, 0]
    y = nerve[:, 1]
    z = nerve[:, 2]

    verts = [list(zip(x, y, z))]

    poly = Poly3DCollection(verts, alpha=0.5)
    color = label_to_color[int(label)]
    poly.set_color(colors.rgb2hex(color))
    poly.set_edgecolor("k")
    ax.add_collection3d(poly)

patch_0 = mpatches.Patch(color=label_to_color[0], label=label_to_str[0], alpha=0.5)
patch_1 = mpatches.Patch(color=label_to_color[1], label=label_to_str[1], alpha=0.5)
plt.legend(handles=[patch_0, patch_1], prop={"size": 20})

plt.show()