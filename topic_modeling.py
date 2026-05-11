import pandas
import numpy
import os
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary
from tqdm import tqdm

# print top-words per topic for overview
def print_topics(model, feature_names, n_top_words=10):
    for topic_index, topic in enumerate(model.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
        print(f"  Thema {topic_index + 1}: {', '.join(top_words)}")

# extract top-words as list for coherence score
def get_top_words_list(model, feature_names, n_top_words=10):
    return [
        [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
        for topic in model.components_
    ]

# calc coherence score
def compute_coherence(top_words_list, tokenized_texts, dictionary):
    coherence_model = CoherenceModel(
        topics=top_words_list,
        texts=tokenized_texts,
        dictionary=dictionary,
        coherence='c_v'
    )
    return coherence_model.get_coherence()

# save topics & scores for external view 
def save_topics(model, feature_names, filename, n_top_words=10):
    with open(f"results/{filename}", "w", encoding="utf-8") as f:
        f.write("Leere oder fehlende Texte, sowie anonymisierte Platzhalter (z.Bsp. 'xxxx') wurden vor der weiteren Verarbeitung entfernt, da sie zu Problemen bei der Vektorisierung führen können.\n\n")
        for topic_index, topic in enumerate(model.components_):
            top_words = [feature_names[i] for i in topic.argsort()[:-n_top_words -1:-1]]
            f.write(f"Topic {topic_index + 1}: {', '.join(top_words)}\n")

def save_scores(nmf_score, lda_score):
    with open("results/coherence_scores.txt", "w", encoding="utf-8") as f:
        f.write(f"NMF Coherence Score: {nmf_score:.4f}\n")
        f.write(f"LDA Coherence Score: {lda_score:.4f}\n")
        f.write(f"Better Model: {'NMF' if nmf_score > lda_score else 'LDA'}")

if __name__ =='__main__':

    # load & clean-up data
    data = pandas.read_csv("data/complaints_clean.csv")
    data = data.dropna(subset=['clean_text'])
    data = data[data['clean_text'].str.strip() != ""]
    texts = data['clean_text'].dropna().astype(str).tolist()

    # tokenize texts for coherence score
    tokenized_texts = [text.split() for text in tqdm(texts, desc="Tokenisierung")]

    # craft gensim dict for coherence score 
    dictionary = Dictionary(tokenized_texts)

    # create directory for results (if not already exists)
    os.makedirs("results", exist_ok=True)

    # TF-IDF for NMF
    tfidf = TfidfVectorizer(max_features=1000, min_df=5, max_df=0.95)
    tfidf_matrix = tfidf.fit_transform(texts)
    feature_names = tfidf.get_feature_names_out()

    # CountVect for LDA
    count_vectorizer = CountVectorizer(max_features=1000, min_df=5, max_df=0.95)
    count_matrix = count_vectorizer.fit_transform(texts)
    count_feature_names = count_vectorizer.get_feature_names_out()

    N_TOPICS = 10

    # method 1: NMF
    nmf_model = NMF(n_components=N_TOPICS, random_state=42, max_iter=400)
    nmf_matrix = nmf_model.fit_transform(tfidf_matrix)
    print("NMF Themen:")
    print_topics(nmf_model, feature_names)
    data['nmf_topic'] = nmf_matrix.argmax(axis=1)
    nmf_top_words = get_top_words_list(nmf_model, feature_names)
    nmf_coherence = compute_coherence(nmf_top_words, tokenized_texts, dictionary)
    print(f"NMF Coherence Score (c_v): {nmf_coherence:.4f}")

    # method 2: LDA
    lda_model = LatentDirichletAllocation(
        n_components=N_TOPICS,
        random_state=42,
        max_iter=10,
        learning_method='online'
    )
    lda_matrix = lda_model.fit_transform(count_matrix)
    print("LDA Themen:")
    print_topics(lda_model, count_feature_names)
    data['lda_topic'] = lda_matrix.argmax(axis=1)
    lda_top_words = get_top_words_list(lda_model, count_feature_names)
    lda_coherence = compute_coherence(lda_top_words, tokenized_texts, dictionary)
    print(f"LDA Coherence Score (c_v): {lda_coherence:.4f}")

    # comparison method 1 & 2
    print("\nVergleich Methode 1: NMF vs Methode 2: LDA")
    print(f"NMF Coherence Score: {nmf_coherence:.4f}")
    print(f"LDA Coherence Score: {lda_coherence:.4f}")
    print(f"Bessere Methode: {'NMF' if nmf_coherence > lda_coherence else 'LDA'} (höherer Score = besser interpretierbare Themen)")

    print(f"\nNMF Themenverteilung (Std): {data['nmf_topic'].value_counts().std():.0f}")
    print(f"LDA Themenverteilung (Std): {data['lda_topic'].value_counts().std():.0f}")
    print("(Niedrigere Std = gleichmäßigere Themenverteilung)")

    overlaps = []
    for nmf_top in tqdm(nmf_top_words, desc="Überschneidungsanalyse"):
        for lda_top in lda_top_words:
            overlaps.append(len(set(nmf_top) & set(lda_top)))

    avg_overlap = numpy.mean(overlaps)
    print(f"Durchschnittliche Wortüberschneidung NMF/LDA: {avg_overlap:.2f} von 10 Wörtern")
    with open("results/overlap.txt", "w", encoding="utf-8") as f: 
        f.write(f"Average overlap NMF/LDA: {avg_overlap:.2f} of 10 words")

    data.to_csv("data/complaints_topics.csv", index=False)

    save_topics(nmf_model, feature_names, "nmf_topics.txt")
    save_topics(lda_model, count_feature_names, "lda_topics.txt")
    save_scores(nmf_coherence, lda_coherence)

    print("\nErgebnisse gespeichert, Themenextraktion abgeschlossen! Final step done")