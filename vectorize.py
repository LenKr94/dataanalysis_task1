import pandas
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
import numpy as np
from tqdm import tqdm

def document_vector(tokens, model):
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(model.vector_size)
    
if __name__ == '__main__':

    # load data
    data = pandas.read_csv("data/complaints_clean.csv")
    texts = data['clean_text'].dropna().tolist()

    # vectorizing, method 1: TF-IDF
    tfidf = TfidfVectorizer(max_features=1000)
    tfidf_matrix = tfidf.fit_transform(texts)
    print(f"TF-IDF Matrix Form: {tfidf_matrix.shape}")
    print(f"Beispiel Top-10 Terme: {tfidf.get_feature_names_out()[:10]}")

    # vectorizing, method 2: Word2Vec Embeddings
    tokenized_texts = [text.split() for text in tqdm(texts, desc="Tokenisierung")]
    w2v_model = Word2Vec(
        sentences=tqdm(tokenized_texts, desc="Word2Vec Training"),
        vector_size=100,
        window=5,
        min_count=2,
        workers=4
    )
    w2v_matrix = np.array([
        document_vector(text.split(), w2v_model)
        for text in tqdm(texts, desc="Dokumentvektoren")
    ])
    print(f"Word2Vec Matrix Form: {w2v_matrix.shape}")

    # comparison method 1 & 2
    print("\nVergleich Methode 1: TF-IDF vs Methode 2: Word2Vec")
    print(f"TF-IDF:   {tfidf_matrix.shape[0]} Dokumente, {tfidf_matrix.shape[1]} Features (sparse)")
    print(f"Word2Vec: {w2v_matrix.shape[0]} Dokumente, {w2v_matrix.shape[1]} Features (dense)")
    print(f"\nTF-IDF Sparsity: {(1 - tfidf_matrix.nnz / (tfidf_matrix.shape[0] * tfidf_matrix.shape[1])):.2%} der Werte sind 0")
    print(f"Word2Vec Wertebereich: min={w2v_matrix.min():.4f}, max={w2v_matrix.max():.4f}")

    print("\nWord2Vec Semantische Ähnlichkeit (Beispiel)")
    test_words = ['credit', 'loan', 'bank', 'payment']
    for word in test_words:
        if word in w2v_model.wv:
            similar = w2v_model.wv.most_similar(word, topn=3)
            print(f"'{word}' ähnlich zu: {similar}")

    np.save("data/w2v_matrix.npy", w2v_matrix)

    print("\nErgebnisse gespeichert, Vektorisierung abgeschlossen! Ready for topic_modeling.py")