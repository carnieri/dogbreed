from fastai.vision.all import *
from itertools import zip_longest


def label_func(x):
    return x.parent.name


def normalize_vector(X_, inplace=False):
    """L2-normalize the vector."""
    if inplace:
        X = X_
    else:
        X = X_.copy()
    norm = np.linalg.norm(X)
    if norm == 0:
        return X
    return X / norm


def normalize_rows(X_, inplace=False):
    """Normalize each row of a matrix independently.
    Changes can be made inplace if specified.
    Rows with only zeros are skipped to avoid division by zero.
    """
    if inplace:
        X = X_
    else:
        X = X_.copy()
    if len(X.shape) == 1:
        return normalize_vector(X)
    else:
        norm = np.linalg.norm(X, axis=1)
        nonzero = norm != 0
        X[nonzero, :] /= norm[nonzero, None]
        return X


def slice_model(original_model, from_layer=None, to_layer=None):
    return nn.Sequential(*list(original_model.children())[from_layer:to_layer])


def get_embedding_from_batch(learn, embedder, imgs):
    """img_paths should contains at most bs (batch size) images.
    returns: [len(img_paths), embedding_size] Tensor"""
    x_ = learn.dls.test_dl(imgs)
    (x,) = first(x_)
    with torch.inference_mode():
        y = embedder(x)
    embeddings = y.squeeze().detach().cpu().numpy()
    es = normalize_rows(embeddings)
    return es


def grouper(iterable, n, fillvalue=None):
    "Collect data into non-overlapping fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)


def get_embedding(learn, embedder, imgs, bs=64):
    embeddings = []
    bs = 64
    for batch in grouper(imgs, bs):
        # remove filler values (may appear in last batch)
        batch = [x for x in batch if x is not None]
        es = get_embedding_from_batch(learn, embedder, batch)
        embeddings.extend(es)
    embeddings = np.array(embeddings)
    if len(embeddings.shape) == 1:
        embeddings = np.expand_dims(embeddings, axis=0)
    return embeddings


def get_embedding_from_paths(learn, embedder, img_paths):
    """img_paths should contains at most bs (batch size) images.
    returns: [len(img_paths), embedding_size] Tensor"""
    # print(f"get_embedding_from_paths: received {len(img_paths)} paths")
    imgs = [PILImage.create(p) for p in img_paths]
    return get_embedding(learn, embedder, imgs)
