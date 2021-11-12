import numpy as np
import faiss
from utils import slice_model, get_embedding


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

    # def enroll(self, img, img_class):
    #     es = self.get_embedding(img)
    #     TO BE CONTINUED

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
        class_names = [self.id_to_class_name[self.ids[ix]] for ix in neighbors]
        return distances, neighbors, class_names

    def search(self, query_img, k=5):
        e = self.get_embedding([query_img])
        return self.search_from_vector(e, k=k)


class NaiveImageSearch:
    def __init__(self, learn, n_features=2048):
        self.learn = learn
        self.embedder = slice_model(self.learn.model, to_layer=-1)

        self.n_features = n_features
        self.embeddings = np.zeros((0, self.n_features), dtype=np.float32)
        self.ids = []
        self.class_name_to_id = {}
        self.id_to_class_name = {}
        self.max_id = 0

        self.imgs = []  # for debugging only

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

    def enroll(self, img, class_name):
        e = self.get_embedding([img])[0, :]
        class_id = self.get_id(class_name)
        self.embeddings.append(e)
        self.ids.append(class_id)

    def enroll_many(self, imgs, class_names):
        n_samples = len(imgs)
        embeddings = self.get_embedding(imgs)
        print(f"enroll_many es.shape: {embeddings.shape}")
        for i in range(n_samples):
            class_id = self.get_id(class_names[i])
            self.ids.append(class_id)
        self.embeddings = np.vstack((self.embeddings, embeddings))
        self.imgs.extend(imgs)

    def search(self, query_img, k=1):
        e = self.get_embedding([query_img])
        # use squared distance instead of distance, for speed
        d = ((self.embeddings - e) ** 2).sum(axis=1)
        d_list = d.tolist()
        sorted_pairs = sorted(zip(d_list, range(len(d))))
        winning_distances, winning_ixs = zip(*sorted_pairs)
        winning_class_names = [
            self.id_to_class_name[self.ids[ix]] for ix in winning_ixs
        ]
        k = min(k, len(sorted_pairs))
        return (
            winning_distances[:k],
            winning_ixs[:k],
            winning_class_names[:k],
        )
