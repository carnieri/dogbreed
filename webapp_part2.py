from pathlib import Path

import streamlit as st
from fastai.vision.all import *
from fastai.vision.widgets import *

from search import FaissImageSearch
import faiss
from utils import label_func


def read_model(path="models/exported_resnext50_32x4d.pickle"):
    learn = load_learner(Path(path))
    return learn


def load_faiss_index(path_to_faiss="models/faiss_index.pickle"):
    """Load and deserialize the Faiss index."""
    if Path(path_to_faiss).exists():
        fd = open(path_to_faiss, "rb")
        data = pickle.load(fd)
        fd.close()
        return faiss.deserialize_index(data)
    else:
        return None


def load_FaissImageSearch(path="models"):
    return FaissImageSearch.load(path)


def get_image_from_upload():
    uploaded_file = st.file_uploader("Upload Files", type=["png", "jpeg", "jpg"])
    if uploaded_file is not None:
        return PILImage.create((uploaded_file))
    return None


def get_prediction(searcher, img, threshold):
    if st.button("Classify"):
        ds, ixs, names, winner = searcher.search(img, threshold=threshold)
        st.write(f"distances: {ds}")
        # st.write(f"ixs: {ixs}")
        st.write(f"breeds: {names}")
        if winner is None:
            winner = "Unknown"
        st.write(f"**Prediction: {winner}**")
    else:
        st.write(f"Click the button to classify")


def pre_enroll(searcher, path="dogs/recognition/enroll/"):
    enroll_paths = get_image_files(path)
    enroll_class_names = [label_func(p) for p in enroll_paths]
    enroll_imgs = [PILImage.create(p) for p in enroll_paths]
    st.write("Pre-enrolling images from 20 dog breeds, this may take a few minutes...")
    searcher.enroll_many(enroll_imgs, enroll_class_names)
    st.write("Done pre-enrolling.")


def main():
    st.title("Dog breed search with index")

    # Load model and searcher
    searcher = load_FaissImageSearch("models")
    if searcher is not None:
        st.write(f"Loaded index from pickle")
    else:
        model = read_model()
        searcher = FaissImageSearch(model)
        pre_enroll(searcher)
        searcher.dump("models")

    img = get_image_from_upload()
    if img is not None:
        st.image(img.to_thumb(500, 500), caption="Uploaded Image")
        threshold = float(
            st.text_input("Distance threshold (recommended: 0.78)", "0.78")
        )
        get_prediction(searcher, img, threshold)


if __name__ == "__main__":
    main()
