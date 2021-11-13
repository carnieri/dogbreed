from fastai.vision.all import *
import numpy as np
import faiss

from utils import label_func, get_embedding, slice_model


class FaissImageSearch:
    def __init__(self, learn, n_features=2048, metric="cosine"):
        self.learn = learn
        self.embedder = slice_model(self.learn.model, to_layer=-1)
        self.ids = []  # class id of each instance that is added
        self.class_name_to_id = {}
        self.id_to_class_name = {}
        self.max_id = 0

        self.imgs = []  # for debugging only

        self.n_features = n_features
        if metric == "euclidean":
            self.index = faiss.IndexFlatL2(self.n_features)
        elif metric == "cosine":
            self.index = faiss.IndexFlatIP(self.n_features)

    def get_embedding(self, imgs):
        return get_embedding(self.learn, self.embedder, imgs)

    def get_id(self, class_name):
        if class_name in self.class_name_to_id:
            class_id = self.class_name_to_id[class_name]
        else:
            class_id = self.max_id
            self.class_name_to_id[class_name] = class_id
            self.id_to_class_name[class_id] = class_name
            self.max_id += 1
        return class_id

    def enroll_many(self, imgs, class_names):
        embeddings = self.get_embedding(imgs)
        print(f"enroll_many embeddings.shape: {embeddings.shape}")
        ids = np.array([self.get_id(class_name) for class_name in class_names])
        self.ids.extend(ids)
        self.imgs.extend(imgs)
        self.index.add(embeddings)

    def search_from_vector(self, e, k=5):
        assert e.shape == (1, self.n_features)
        distances, neighbors = self.index.search(e, k)
        # we always work with a single query item
        distances = distances[0]
        neighbors = neighbors[0]
        class_ids = np.array([self.ids[ix] for ix in neighbors if ix != -1])
        class_names = [
            self.id_to_class_name[self.ids[ix]] for ix in neighbors if ix != -1
        ]
        if class_names == []:
            return distances, neighbors, class_names, None

        max_class = max(set(class_names), key=class_names.count)
        max_id = self.class_name_to_id[max_class]
        count_max_class = class_names.count(max_class)
        if count_max_class > k // 2:
            # we have a winner
            # collect distances from winning class
            distances_max_class = distances[class_ids == max_id]
            mean_dist = np.mean(distances_max_class)
            # print(f"mean_dist: {mean_dist}")
            # TODO filter output by distance threshold
            return distances, neighbors, class_names, max_class
        else:
            # not confident enough to classify query
            return distances, neighbors, class_names, None

    def search(self, query_img, k=5):
        e = self.get_embedding([query_img])
        return self.search_from_vector(e, k=k)


def search_from_path(searcher, path, k=3):
    img = PILImage.create(path)
    ds, ixs, names, winner = searcher.search(img, k=k)
    return ds, ixs, names, winner


def search_accuracy(searcher, imgs, embeddings, class_names, k=5, threshold=0.0):
    dist_correct = []
    dist_incorrect = []
    dist_empty = []
    dist_all = []
    correct = 0
    incorrect = 0
    for i in range(len(imgs)):
        # if i % 100 == 0:
        #     print(i)
        ds, ixs, names, winner = searcher.search_from_vector(
            np.expand_dims(embeddings[i], axis=0), k=k
        )
        if winner is not None:
            if winner == class_names[i] and ds[0] >= threshold:
                correct += 1
                dist_correct.append(ds[0])
            else:
                incorrect += 1
                dist_incorrect.append(ds[0])
        else:
            dist_empty.append(ds[0])
        dist_all.append(ds[0])
    acc = float(correct) / len(imgs)
    dist_all = np.array(dist_all)
    dist_correct = np.array(dist_correct)
    dist_incorrect = np.array(dist_incorrect)
    dist_empty = np.array(dist_empty)
    return acc, dist_all, dist_correct, dist_incorrect, dist_empty


def plot_results(imgs, embeddings, class_names):
    (
        acc,
        dist_all,
        dist_correct,
        dist_incorrect,
        dist_empty,
    ) = search_accuracy(imgs, embeddings, class_names)
    print(f"dist_all.shape: {dist_all.shape}")
    print(f"dist_correct.shape: {dist_correct.shape}")
    print(f"dist_incorrect.shape: {dist_incorrect.shape}")
    print(f"dist_empty.shape: {dist_empty.shape}")
    plt.subplot(4, 1, 1)
    plt.plot(dist_all)
    plt.axis([0, len(dist_all), 0, 1.0])
    plt.show()

    plt.subplot(4, 1, 2)
    plt.plot(dist_correct)
    plt.axis([0, len(dist_correct), 0, 1.0])
    plt.show()

    plt.subplot(4, 1, 3)
    plt.plot(dist_incorrect)
    plt.axis([0, len(dist_incorrect), 0, 1.0])
    plt.show()

    plt.subplot(4, 1, 4)
    plt.plot(dist_empty)
    plt.axis([0, len(dist_empty), 0, 1.0])
    plt.show()


def calculate_rejection_accuracy(searcher, imgs, embeddings, threshold=0.78):
    """Assumes that samples are from classes unknown to the database."""
    correct = 0
    for i in range(len(imgs)):
        if i % 100 == 0:
            print(i)
        ds, ixs, names, winner = searcher.search_from_vector(
            np.expand_dims(embeddings[i], axis=0), k=1
        )
        if ds[0] < threshold:
            correct += 1
    acc = float(correct) / len(imgs)
    print(f"acc: {acc}")
    return acc
