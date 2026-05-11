# DLBDSEDE02_D Projekt: Data Analysis
## Task 1: NLP-Analyse von Beschwerdetexten

Dieses Repository enthält den Code zur NLP-Analyse der CFPB Consumer Complaint Database.
Ziel ist die Extraktion häufig vorkommender Themen aus unstrukturierten Beschwerdetexten.

**Hinweis:** Zur Kontrolle werden während der Laufzeit des jeweiligen Skripts Daten & Progressbars im Terminal ausgegeben. Die finalen Ergebnisse werden zusätzlich im Ordner '/results' gespeichert & sind ohne Ausführung der Skripte einsehbar.

## Datensatz
- **Quelle:** [CFPB Consumer Complaint Database](https://www.consumerfinance.gov/data-research/consumer-complaints/)
    - kostenfreier download, verfügbar u.A. als .csv 
- **Umfang:** ~14 Millionen Einträge, davon 50.000 als Stichprobe verwendet um lange Laufzeiten zu vermeiden
- **Relevante Spalte:** `consumer_complaint_narrative`

## Projektstruktur
```
dataanalysis_task1/
│
├── load_data.py          # Datensatz laden und erste Übersicht
├── prepare_data.py       # Textvorverarbeitung und Bereinigung
├── vectorize.py          # Vektorisierung (gewählte Methoden: TF-IDF & Word2Vec)
├── topic_modeling.py     # Themenextraktion (gewählte Methoden: NMF & LDA)
├── requirements.txt      # Alle verwendeten Bibliotheken
├── results/              # gespeicherte Ergebnisse (Topics, Scores, Overlap)
└── data/                 # Datensatz (lokal, nicht im Repository; kostenfrei download via o.g. Quelle)
```

## Installation
```bash
# Virtuelle Umgebung erstellen und aktivieren
python -m venv nlp_env
nlp_env\Scripts\activate

# Abhängigkeiten installieren
pip install -r requirements.txt

```

## Ausführung
Die Skripte werden in folgender Reihenfolge ausgeführt:
```bash
python load_data.py
python prepare_data.py
python vectorize.py
python topic_modeling.py
```

## Verwendete Bibliotheken & deren Verwendugnszweck
| Bibliothek | Verwendungszweck |
|---|---|
| Pandas, NumPy | Datenverarbeitung |
| NLTK | Textvorverarbeitung |
| scikit-learn | TF-IDF, NMF, LDA |
| Gensim | Word2Vec Embeddings |
| tqdm | Fortschrittsanzeige |

## Methoden
### Textvorverarbeitung mittels NLTK (prepare_data.py)
Kleinschreibung, Entfernung von Sonderzeichen, URLs und Zahlen, Tokenisierung, Stoppwortentfernung und Lemmatisierung.
**Hinweis:** Leere oder fehlende Texte, sowie anonymisierte Platzhalter (z.Bsp. 'xxxx') wurden vor der weiteren Verarbeitung entfernt, da sie zu Problemen bei der Vektorisierung führen können. Dies führt dazu, dass nicht mehr alle der 50.000 Einträge verwendet werden. 

### Vektorisierung (vectorize.py)
- **Methode 1: TF-IDF** 
- **Methode 2: Word2Vec** 

### Themenextraktion (topic_modeling.py)
Es werden bewusst jeweils 10 Topics erzeugt, um eine ausreichend differenzierte Themenstruktur zu erhalten.
- **Methode 1: NMF** 
- **Methode 2: LDA** 

## Ergebnisse
Alle Ergebnisse werden nach Ausführung der Skripte automatisch im Ordner '/results' gespeichert & können ohne Ausführung der Skripte eingesehen werden. 

### Inhalt der Ergebnisdaten
- **'nmf_topics.txt': 10 extrahierte Topics (NMF)**
- **'lda_topics.txt': 10 extrahierte Topics (LDA)**
- **'coherence_scores.txt': Vergleich der Modellqualität (Coherence Scores)**
- **'overlap.txt': durchschnittliche Wortüberschneidung der Methoden**

### Themen-Beispiele
Die extrahierten Themen umfassen u.A.: 
- **Kreditkartenberichte & fehlerhafte Einträge**
- **Ident-Diebstahl & Betrug**
- **Debt Collection & Validierungsanfragen**
- **Zahlungsprobleme & Gebühren**
- **Banking- & Kontoprobleme**

**Hinweis:** Ein Teil der Tokens enthält anonymisierte Platzhalter (z.B. "xxxx"), die aus dem Datensatz stammen. 