import os
from Bio import SeqIO
from gensim.models import Word2Vec
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# Get script directory
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    script_dir = os.path.abspath(".")
    print(f"Warning: __file__ not defined. Using current directory: {script_dir}")


# -----------------------------
# Helper Functions
# -----------------------------

def load_fasta(file_path):
    """Loads all sequences from a single FASTA file. Returns list of (id, sequence)."""
    sequences = []
    try:
        for record in SeqIO.parse(file_path, "fasta"):
            sequences.append((record.id, str(record.seq).upper()))
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    return sequences


def get_kmers(seq, k=3):
    return [seq[i:i + k] for i in range(len(seq) - k + 1)]


def train_seq2vec(sequences_with_ids, k=3, vector_size=100, window=5, min_count=1, sg=1):
    seq_list = [seq for sid, seq in sequences_with_ids]
    sentences = [get_kmers(seq, k) for seq in seq_list]
    print(f"Training Word2Vec on {len(sentences)} sequences...")
    return Word2Vec(sentences, vector_size=vector_size, window=window, min_count=min_count, sg=sg)


def seq_to_vectors_list(seq, model, k=3):
    kmers = get_kmers(seq, k)
    vectors = [model.wv[k] for k in kmers if k in model.wv]

    if len(vectors) == 0:
        return [np.zeros(model.vector_size)]
    return vectors


# -----------------------------
# Sliding Windows
# -----------------------------

def windows_from_vectors(vector_list, window_size=512, stride=256):
    windows = []
    n = len(vector_list)

    if n < window_size:
        padded = vector_list + [np.zeros_like(vector_list[0])] * (window_size - n)
        return [padded]

    for start in range(0, n - window_size + 1, stride):
        windows.append(vector_list[start:start + window_size])

    # ensure last window included
    if (n - window_size) % stride != 0:
        windows.append(vector_list[-window_size:])

    return windows


# -----------------------------
# Main Script
# -----------------------------

if __name__ == "__main__":

    base_raw_path = os.path.join(script_dir, "raw")

    FILE_LABEL_MAP = {

    # HIV (env)
    "hiv_clade_B_env.fasta": "hiv_clade_b",
    "hiv_clade_C_env.fasta": "hiv_clade_c",
    

    # SARS-CoV-2 (Spike)
    "sars_omicron_S.fasta": "sars_omicron",
    "sars_delta_S.fasta": "sars_delta",
    

    # Dengue (E)
   
    "denv1_genotype_IV_E.fasta": "dengue_genotype_iv",
    "denv1_genotype_V_E.fasta":  "dengue_genotype_v",

    # Hepatitis B (S / HBsAg)
    
    "hbv_genotype_C_S.fasta": "hepatitis_b_genotype_c",
    "hbv_genotype_D_S.fasta": "hepatitis_b_genotype_d",

    # Hepatitis C (E2)
    "hcv_genotype_1a_E2.fasta": "hepatitis_c_genotype_1a",
    "hcv_genotype_3_E2.fasta":  "hepatitis_c_genotype_3",

    # Measles (H)
    "measles_genotype_B3_H.fasta": "measles_b3",
    "measles_genotype_D8_H.fasta": "measles_d8",
    "measles_genotype_H1_H.fasta": "measles_h1",

    # Influenza A (HA)
    "influenza_A_H1N1_HA.fasta": "influenza_a_h1n1",
    "influenza_A_H3N2_HA.fasta": "influenza_a_h3n2",
    "influenza_A_H5N1_HA.fasta": "influenza_a_h5n1",
}


    OUTPUT_DIR = os.path.join(script_dir, "processed_3way_split")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    TEST_SPLIT = 0.20
    VAL_SPLIT = 0.10       # out of REMAINING 80%
    KMER_SIZE = 3
    VECTOR_DIM = 100

    # -----------------------------
    # Phase 1 — Load sequences
    # -----------------------------
    print("\n--- Loading All Sequences ---")

    all_sequences = []
    all_labels = []

    for filename, label in FILE_LABEL_MAP.items():
        fpath = os.path.join(base_raw_path, filename)
        print(f"Loading {label} from {filename}...")
        seqs = load_fasta(fpath)

        for sid, seq in seqs:
            all_sequences.append((sid, seq, label))
            all_labels.append(label)

    print(f"Total sequences loaded: {len(all_sequences)}")


    # -----------------------------
    # Phase 2 — 3-way split (sequence level)
    # -----------------------------
    print("\n--- Creating Train / Val / Test Split (SEQUENCE level) ---")

    train_seqs, test_seqs = train_test_split(
        all_sequences,
        test_size=TEST_SPLIT,
        stratify=all_labels,
        random_state=42
    )

    # Prepare labels for validation split
    train_labels_only = [lbl for sid, seq, lbl in train_seqs]

    train_seqs, val_seqs = train_test_split(
        train_seqs,
        test_size=VAL_SPLIT / (1 - TEST_SPLIT),
        stratify=train_labels_only,
        random_state=42
    )

    print(f"Train sequences: {len(train_seqs)}")
    print(f"Validation sequences: {len(val_seqs)}")
    print(f"Test sequences: {len(test_seqs)}")


    # -----------------------------
    # Phase 3 — Train Word2Vec only on training sequences
    # -----------------------------
    print("\n--- Training Word2Vec on Training Sequences ---")

    train_for_w2v = [(sid, seq) for sid, seq, lbl in train_seqs]
    w2v_model = train_seq2vec(train_for_w2v, k=KMER_SIZE, vector_size=VECTOR_DIM)
    word2vec_dir = os.path.abspath(os.path.join(script_dir, "..", "models"))
    os.makedirs(word2vec_dir, exist_ok=True)
    w2v_model.save(os.path.join(word2vec_dir, "word2vec.model"))

    print("Saved Word2Vec model.")


    # -----------------------------
    # Phase 4 — Convert sequences to windows
    # -----------------------------
    def process_set(sequence_list, name):
        print(f"\nProcessing {name} set...")

        windows_out = []
        rows = []

        for sid, seq, label in sequence_list:
            vec_list = seq_to_vectors_list(seq, w2v_model, k=KMER_SIZE)
            windows = windows_from_vectors(vec_list, window_size=512, stride=256)

            for w in windows:
                windows_out.append(w)
                rows.append({
                    "sequence_id": sid,
                    "label": label,
                    "original_sequence_length": len(seq)
                })

        windows_out = np.array(windows_out, dtype=np.float32)

        np.save(os.path.join(OUTPUT_DIR, f"{name}_embeddings.npy"), windows_out)
        pd.DataFrame(rows).to_csv(os.path.join(OUTPUT_DIR, f"{name}_labels.csv"), index=False)

        print(f"Saved {name}_embeddings.npy → shape {windows_out.shape}")
        print(f"Saved {name}_labels.csv → {len(rows)} rows")

    process_set(train_seqs, "train")
    process_set(val_seqs, "val")
    process_set(test_seqs, "test")

    print("\nAll files created successfully with CLEAN 3-way split!")


# ======================================
# CHECK CLASS DISTRIBUTIONS
# ======================================
print("\n======= CLASS DISTRIBUTION CHECK =======")

def show_distribution(csv_path, name):
    df = pd.read_csv(csv_path)
    print(f"\n{name} distribution (rows = window samples):")
    print(df['label'].value_counts())
    print("\nUnique sequence IDs:", df['sequence_id'].nunique())

show_distribution(os.path.join(OUTPUT_DIR, "train_labels.csv"), "TRAIN")
show_distribution(os.path.join(OUTPUT_DIR, "val_labels.csv"), "VALIDATION")
show_distribution(os.path.join(OUTPUT_DIR, "test_labels.csv"), "TEST")

print("\n========================================")
