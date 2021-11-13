import numpy as np
import faiss
from utils import slice_model, get_embedding


class FaissImageSearch:
    def __init__(self, learn, n_features=2048, metric="cosine"):
        self.learn = learn
        self.embedder = slice_model(self.learn.model, to_layer=-1)
        print("learn.model:")
        print(self.learn.model)
        print("embedder:")
        print(self.embedder)
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
