import os
import sys
import numpy as np
import joblib
import tensorflow as tf
from gensim.models import Word2Vec
from Bio import SeqIO
from Bio.Align import PairwiseAligner
from tensorflow.keras.preprocessing.sequence import pad_sequences
import warnings

# =====================================================================
#                        MONKEY PATCH FOR LSTM FIX
# =====================================================================
# Allows loading old TF 2.x LSTM models in TF 2.20+
from keras.layers import LSTM, GRU

def patched_init(self, *args, **kwargs):
    invalid = ["time_major", "implementation", "go_backwards"]
    for arg in invalid:
        if arg in kwargs:
            kwargs.pop(arg, None)
    original_init(self, *args, **kwargs)

original_init = LSTM.__init__
LSTM.__init__ = patched_init

original_gru_init = GRU.__init__
GRU.__init__ = patched_init

# =====================================================================
#                             SUPPRESS WARNINGS
# =====================================================================
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')
warnings.filterwarnings('ignore')

# =====================================================================
#                               PATH SETUP
# =====================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))               # Identify_mutation/
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))       # BIO-INFORMATICS/

REFERENCES_DIR = os.path.join(BASE_DIR, "references")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "results5")

W2V_PATH     = os.path.join(MODEL_DIR, "word2vec.model")
ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.joblib")

# Use your real model filename here:
KERAS_PATH   = os.path.join(MODEL_DIR, "best_model.h5")   # <-- Your working file

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================================================================
#                 MAPPING PREDICTION → REFERENCE FASTA
# =====================================================================
FILE_LABEL_MAP = {
    "hiv_clade_B_env.fasta": "hiv_clade_b",
    "hiv_clade_C_env.fasta": "hiv_clade_c",

    "sars_omicron_S.fasta": "sars_omicron",
    "sars_delta_S.fasta": "sars_delta",

    "denv1_genotype_IV_E.fasta": "dengue_genotype_iv",
    "denv1_genotype_V_E.fasta": "dengue_genotype_v",

    "hbv_genotype_C_S.fasta": "hepatitis_b_genotype_c",
    "hbv_genotype_D_S.fasta": "hepatitis_b_genotype_d",

    "hcv_genotype_1a_E2.fasta": "hepatitis_c_genotype_1a",
    "hcv_genotype_3_E2.fasta": "hepatitis_c_genotype_3",

    "measles_genotype_B3_H.fasta": "measles_b3",
    "measles_genotype_D8_H.fasta": "measles_d8",
    "measles_genotype_H1_H.fasta": "measles_h1",

    "influenza_A_H1N1_HA.fasta": "influenza_a_h1n1",
    "influenza_A_H3N2_HA.fasta": "influenza_a_h3n2",
    "influenza_A_H5N1_HA.fasta": "influenza_a_h5n1",
}

# =====================================================================
#                           WINDOW + KMER FUNCTIONS
# =====================================================================
KMER_SIZE = 3
MAX_SEQUENCE_LENGTH = 512
WINDOW_STRIDE = 256

def get_kmers(seq, k=KMER_SIZE):
    return [seq[i:i+k] for i in range(len(seq)-k+1)]

def sliding_windows(seq, win=MAX_SEQUENCE_LENGTH, stride=WINDOW_STRIDE):
    for i in range(0, len(seq)-win+1, stride):
        yield seq[i:i+win]
    if len(seq) < win:
        yield seq

def seq_to_window_vectors(seq, w2v):
    out = []
    for window in sliding_windows(seq):
        kmers = get_kmers(window)
        vectors = [w2v.wv[k] for k in kmers if k in w2v.wv]
        if not vectors:
            vectors = [np.zeros(w2v.vector_size)]
        out.append(vectors)
    return out

# =====================================================================
#                           VARIANT PREDICTION
# =====================================================================
def predict_variant(fasta_path):
    print("[INFO] Loading ML models...")
    print(f"[INFO] Word2Vec: {W2V_PATH}")
    print(f"[INFO] Encoder:  {ENCODER_PATH}")
    print(f"[INFO] Classifier: {KERAS_PATH}")

    w2v = Word2Vec.load(W2V_PATH)
    encoder = joblib.load(ENCODER_PATH)

    # model loading with patched LSTM
    model = tf.keras.models.load_model(KERAS_PATH, compile=False)

    print("[INFO] All models loaded.\n")

    predictions = {}

    for record in SeqIO.parse(fasta_path, "fasta"):
        seq_id = record.id
        seq = str(record.seq).upper()

        windows = seq_to_window_vectors(seq, w2v)
        preds = []

        for win in windows:
            X = pad_sequences([win], maxlen=MAX_SEQUENCE_LENGTH,
                              dtype="float32", padding="post", truncating="post")
            prob = model.predict(X, verbose=0)[0]
            preds.append(prob)

        avg = np.mean(preds, axis=0)
        cls = np.argmax(avg)
        label = encoder.inverse_transform([cls])[0]

        predictions[seq_id] = label

    return predictions

# =====================================================================
#                           REFERENCE HANDLING
# =====================================================================
def get_reference_from_prediction(pred):
    for fname, lbl in FILE_LABEL_MAP.items():
        if lbl == pred:
            ref_path = os.path.join(REFERENCES_DIR, fname)
            if not os.path.exists(ref_path):
                raise FileNotFoundError(f"Missing reference: {ref_path}")
            return ref_path
    raise ValueError(f"No reference mapped for prediction: {pred}")

def load_fasta(path):
    return str(next(SeqIO.parse(path, "fasta")).seq)

# =====================================================================
#                       ALIGNMENT + MUTATION FINDER
# =====================================================================
def reconstruct_alignment(aln, a, b):
    ref_blocks, samp_blocks = aln.aligned
    ref_parts, samp_parts = [], []
    rpos = spos = 0

    for (rs, re), (ss, se) in zip(ref_blocks, samp_blocks):
        if rs > rpos:
            ref_parts.append(a[rpos:rs])
            samp_parts.append("-" * (rs - rpos))

        if ss > spos:
            ref_parts.append("-" * (ss - spos))
            samp_parts.append(b[spos:ss])

        ref_parts.append(a[rs:re])
        samp_parts.append(b[ss:se])

        rpos, spos = re, se

    if rpos < len(a):
        ref_parts.append(a[rpos:])
        samp_parts.append("-" * (len(a) - rpos))

    if spos < len(b):
        ref_parts.append("-" * (len(b) - spos))
        samp_parts.append(b[spos:])

    return "".join(ref_parts), "".join(samp_parts)

def find_mutations(ref, samp):
    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 2
    aligner.mismatch_score = -1
    aligner.open_gap_score = -2
    aligner.extend_gap_score = -0.5

    aln = aligner.align(ref, samp)[0]
    ref_aln, samp_aln = reconstruct_alignment(aln, ref, samp)

    mutations = []
    pos = 0

    for r, s in zip(ref_aln, samp_aln):
        if r != "-":
            pos += 1

        if r == s:
            continue
        elif r != "-" and s != "-":
            mutations.append((pos, "SNP", r, s))
        elif r == "-" and s != "-":
            mutations.append((pos, "INSERTION", "-", s))
        elif r != "-" and s == "-":
            mutations.append((pos, "DELETION", r, "-"))

    return mutations

# =====================================================================
#                           OUTPUT WRITER
# =====================================================================
def save_mutations(muts, seq_id, label):
    out = os.path.join(OUTPUT_DIR, f"{seq_id}_{label}_mutations.tsv")
    with open(out, "w") as f:
        f.write("POSITION\tTYPE\tREF\tALT\n")
        for m in muts:
            f.write(f"{m[0]}\t{m[1]}\t{m[2]}\t{m[3]}\n")
    return out

# =====================================================================
#                           FULL PIPELINE
# =====================================================================
def run_pipeline(fasta_path):

    predictions = predict_variant(fasta_path)

    for seq_id, label in predictions.items():

        print(f"\n[INFO] Prediction: {seq_id} → {label}")

        ref_path = get_reference_from_prediction(label)
        print(f"[INFO] Using reference: {ref_path}")

        ref = load_fasta(ref_path)
        samp = load_fasta(fasta_path)

        muts = find_mutations(ref, samp)
        out = save_mutations(muts, seq_id, label)

        print(f"[INFO] Mutations saved to: {out}")
        print(f"[INFO] Total found: {len(muts)}")

# =====================================================================
#                                 MAIN
# =====================================================================
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python mutation_pipeline2.py input.fasta")
        sys.exit(1)

    fasta = sys.argv[1]
    run_pipeline(fasta)
