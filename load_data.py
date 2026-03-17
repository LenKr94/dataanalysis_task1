import pandas
from tqdm import tqdm

if __name__ == '__main__':

    # initial task: load data from source
    print("Lese 100.000 Zeilen in Chunks... (kann etwas dauern)")
    chunks = []
    chunk_iter = pandas.read_csv("data/complaints.csv", chunksize=100000)
    for chunk in tqdm(chunk_iter, desc="Einlesen", unit="chunk"):
        chunks.append(chunk)
    data = pandas.concat(chunks, ignore_index=True)

    # 1st step of data-prep: modify column-names
    data.columns = data.columns.str.strip().str.lower().str.replace(" ", "_")

    # 2nd step of data-prep: fast overview
    print(f"Anzahl Beschwerden (Zeilen) & Spalten: {data.shape}")
    print(f"Spalten: {data.columns.tolist()}")
    print(data.head())

    # 3rd step of data-prep: isolate relevant columns
    clean_data = data[["consumer_complaint_narrative", "product", "issue"]].dropna(
        subset=["consumer_complaint_narrative"]
    )

    print(f"Anzahl verwertbarer Beschwerden: {len(clean_data)}")
    
    print("\nVorgang abgeschlossen! Ready for prepare_data.py")