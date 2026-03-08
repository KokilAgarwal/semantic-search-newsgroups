from sklearn.datasets import fetch_20newsgroups

def load_data():

    dataset = fetch_20newsgroups(
        subset='all',
        remove=('headers','footers','quotes')
    )

    texts = dataset.data
    labels = dataset.target

    return texts, labels