import pandas
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm 
tqdm.pandas() 

# initial task: load nltk ressources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt_tab', quiet=True)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# prep text (lowercase, URL removal, punctuation removal, stopwords, lemmatization, remove anonymised words)
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words and not (len(t) > 1 and set(t) == {"x"})]
    return ' '.join(tokens)

if __name__ == '__main__':

    # load data + reduce to avoid endless runtime ;)
    data = pandas.read_csv("data/complaints.csv")
    data.columns = data.columns.str.strip().str.lower().str.replace(" ", "_")
    clean_data = data[["consumer_complaint_narrative", "product", "issue"]].dropna(
        subset=["consumer_complaint_narrative"]
    )
    sample_data = clean_data.sample(n=50000, random_state=42)

    # preprocess data
    tqdm.pandas(desc="Verarbeite Daten")
    sample_data['clean_text'] = sample_data['consumer_complaint_narrative'].progress_apply(preprocess_text)
    sample_data.to_csv("data/complaints_clean.csv", index=False)
    print(f"\n{'Zeilenindex':<12} {'Bereinigter Text'}")
    print("-" * 80)
    for index, text in sample_data['clean_text'].head().items():
        print(f"{index:<12} {text[:65]}...")

    print("\nErgebnisse gespeichert, Vorverarbeitung abgeschlossen! Ready for vectorize.py")