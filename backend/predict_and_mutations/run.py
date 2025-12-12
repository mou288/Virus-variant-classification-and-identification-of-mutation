import os
import numpy as np
import joblib
import tensorflow as tf
from gensim.models import Word2Vec
from Bio import SeqIO
from Bio.Align import PairwiseAligner
from io import StringIO
from tensorflow.keras.preprocessing.sequence import pad_sequences
import warnings

# ========= CLEAN LOGGING =========
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')
warnings.filterwarnings('ignore')

# ========= PATHS =========
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

REFERENCES_DIR = os.path.join(BASE_DIR, "references")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "results")

W2V_PATH = os.path.join(PROJECT_ROOT, "models", "word2vec.model")
ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.joblib")
KERAS_PATH = os.path.join(MODEL_DIR, "best_model.keras")

# ========= LABEL → REFERENCE FASTA =========
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

# ========= CONSTANTS =========
KMER_SIZE = 3
MAX_SEQUENCE_LENGTH = 512
WINDOW_STRIDE = 256


# ========= UTILITIES =========
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


# ========= MAIN PREDICTION FUNCTION (STREAMLIT CALLS THIS) =========
def classify_virus_from_text(fasta_text):
    """
    Takes FASTA text and returns:
    - seq_id
    - predicted label
    """

    fasta_io = StringIO(fasta_text)
    record = next(SeqIO.parse(fasta_io, "fasta"))

    seq_id = record.id
    seq = str(record.seq).upper()

    # Load models once per session
    w2v = Word2Vec.load(W2V_PATH)
    encoder = joblib.load(ENCODER_PATH)
    model = tf.keras.models.load_model(KERAS_PATH, compile=False)

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

    return seq_id, label, seq


def load_reference_sequence(label):
    """Return reference sequence text for mutation comparison."""
    for fname, lbl in FILE_LABEL_MAP.items():
        if lbl == label:
            ref_path = os.path.join(REFERENCES_DIR, fname)
            if not os.path.exists(ref_path):
                raise FileNotFoundError(f"Reference file missing: {ref_path}")
            return str(next(SeqIO.parse(ref_path, "fasta")).seq)

    raise ValueError(f"No reference FASTA mapped for: {label}")

def reconstruct_alignment(aln, a, b):
    """
    Reconstruct full alignment strings (with gaps '-') from PairwiseAligner output.
    """
    ref_blocks, samp_blocks = aln.aligned
    ref_parts, samp_parts = [], []
    rpos = spos = 0

    for (rs, re), (ss, se) in zip(ref_blocks, samp_blocks):
        # gap in sample
        if rs > rpos:
            ref_parts.append(a[rpos:rs])
            samp_parts.append('-' * (rs - rpos))

        # gap in reference
        if ss > spos:
            ref_parts.append('-' * (ss - spos))
            samp_parts.append(b[spos:ss])

        # matched / mismatched segment
        ref_parts.append(a[rs:re])
        samp_parts.append(b[ss:se])

        rpos, spos = re, se

    # tail end of reference
    if rpos < len(a):
        ref_parts.append(a[rpos:])
        samp_parts.append('-' * (len(a) - rpos))

    # tail end of sample
    if spos < len(b):
        ref_parts.append('-' * (len(b) - spos))
        samp_parts.append(b[spos:])

    return "".join(ref_parts), "".join(samp_parts)

def find_mutations(ref: str, samp: str):
    aligner = PairwiseAligner()
    aligner.mode = "global"

    aligner.match_score = 2
    aligner.mismatch_score = -2
    aligner.open_gap_score = -6
    aligner.extend_gap_score = -1

    aln = aligner.align(ref, samp)[0]
    aligned_ref, aligned_samp = reconstruct_alignment(aln, ref, samp)

    mutations = []
    ref_pos = 0
    i = 0
    L = len(aligned_ref)

    while i < L:
        r = aligned_ref[i]
        s = aligned_samp[i]

        # Count reference coordinate
        if r != "-":
            ref_pos += 1

        # -------------------------
        # SNP
        # -------------------------
        if r != "-" and s != "-" and r != s:
            hgvs = f"c.{ref_pos}{r}>{s}"

            mutations.append({
                "type": "SNP",
                "position": ref_pos,
                "ref": r,
                "alt": s,
                "hgvs": hgvs
            })
            i += 1
            continue

        # -------------------------
        # INSERTION
        # Reference has gap, sample has bases
        # -------------------------
        if r == "-" and s != "-":
            pos_before = ref_pos
            ins_bases = []

            while i < L and aligned_ref[i] == "-" and aligned_samp[i] != "-":
                ins_bases.append(aligned_samp[i])
                i += 1

            ins_seq = "".join(ins_bases)

            # HGVS notation: c.(pos_before)_(pos_before+1)insSEQ
            hgvs = f"c.{pos_before}_{pos_before+1}ins{ins_seq}"

            mutations.append({
                "type": "INSERTION",
                "position": pos_before,
                "ref": "-",
                "alt": ins_seq,
                "hgvs": hgvs
            })
            continue

        # -------------------------
        # DELETION
        # Sample has gap, ref has bases
        # -------------------------
        if r != "-" and s == "-":
            del_start = ref_pos
            del_bases = []

            while i < L and aligned_ref[i] != "-" and aligned_samp[i] == "-":
                del_bases.append(aligned_ref[i])
                i += 1
                ref_pos += 1

            del_end = ref_pos - 1
            del_seq = "".join(del_bases)

            # HGVS: c.26delA or c.26_30delAGCTA
            if del_start == del_end:
                hgvs = f"c.{del_start}del{del_seq}"
            else:
                hgvs = f"c.{del_start}_{del_end}del{del_seq}"

            mutations.append({
                "type": "DELETION",
                "position": del_start if del_start == del_end else f"{del_start}-{del_end}",
                "ref": del_seq,
                "alt": "-",
                "hgvs": hgvs
            })
            continue

        i += 1

    return mutations


def mutations_to_tsv_text(muts):
    lines = ["TYPE\tPOSITION\tREF\tALT\tHGVS"]

    for m in muts:
        lines.append(f"{m['type']}\t{m['position']}\t{m['ref']}\t{m['alt']}\t{m['hgvs']}")

    return "\n".join(lines)
