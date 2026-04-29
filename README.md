#  GenomeNet — Viral Variant Classifier & Mutation Analyzer

A deep learning pipeline for classifying viral DNA sequences across **16 classes** and detecting genomic mutations with **HGVS-compliant reporting**. Built with a CNN + Bidirectional LSTM architecture trained on **16,000+ real sequences** fetched from NCBI Entrez.

---

## 📊 Results

| Metric | Score |
|---|---|
| Test Accuracy | **94%** |
| Macro F1-Score | **0.93** |
| Weighted F1-Score | **0.94** |
| Classes | **16** |

### Per-Class F1 Highlights
| Virus | F1 |
|---|---|
| Influenza A (H1N1, H3N2, H5N1) | 1.00 |
| HIV Clade B / C | 0.97 – 0.98 |
| Measles (B3, D8, H1) | 0.98 – 0.99 |
| SARS-CoV-2 (Delta, Omicron) | 0.92 – 0.93 |
| Dengue (Genotype IV, V) | 0.81 – 0.83 |

---

##  Supported Virus Classes

| Virus | Variants |
|---|---|
| HIV-1 | Clade B, Clade C |
| SARS-CoV-2 | Delta, Omicron |
| Dengue Virus | Genotype IV, Genotype V |
| Hepatitis B | Genotype C, Genotype D |
| Hepatitis C | Genotype 1a, Genotype 3 |
| Measles | B3, D8, H1 |
| Influenza A | H1N1, H3N2, H5N1 |

---

##  Architecture

```
FASTA Sequence
      │
      ▼
K-mer Tokenisation (k=3)
      │
      ▼
Word2Vec Embeddings (dim=100)
      │
      ▼
Sliding Window Segmentation (window=512, stride=256)
      │
      ▼
┌─────────────────────────────┐
│  CNN Block 1 (128 filters)  │
│  BatchNorm → MaxPool        │
│  CNN Block 2 (256 filters)  │
│  BatchNorm → MaxPool        │
│  Bidirectional LSTM (128)   │
│  Dense (256) → Softmax (16) │
└─────────────────────────────┘
      │
      ▼
Predicted Variant + Confidence
      │
      ▼
Pairwise Alignment vs Reference
      │
      ▼
HGVS Mutation Report (SNPs, Insertions, Deletions)
```

---

## 📁 Project Structure

```
GenomeNet/
│
├── data/
│   ├── raw/                        # Raw FASTA files from NCBI
│   └── processed_3way_split/       # Preprocessed embeddings (.npy) and labels (.csv)
│
├── models/
│   ├── word2vec.model              # Trained Word2Vec model
│   └── results/
│       ├── best_model.keras        # Best trained CNN+BiLSTM model
│       ├── label_encoder.joblib    # Sklearn label encoder
│       ├── confusion_matrix.png    # Confusion matrix plot
│       ├── accuracy_curve.png      # Training accuracy curve
│       └── test_report.txt         # Full classification report
│
├── src/
│   ├── fetch_sequences.py          # NCBI Entrez data fetching pipeline
│   ├── preprocess.py               # K-mer embedding + sliding window preprocessing
│   ├── train.py                    # CNN + BiLSTM model training
│   └── predict_and_mutations/
│       └── run.py                  # Inference + HGVS mutation detection
│
├── app.py                          # Streamlit web application
├── requirements.txt
└── README.md
```

---

## ⚙️ How It Works

### 1. Data Collection
Sequences are fetched directly from **NCBI Entrez** using virus-specific search queries with length and quality filters (max 3% ambiguous bases). Up to 1,000 sequences per variant are sampled for class balance.

### 2. Preprocessing
- Each sequence is tokenised into overlapping **3-mers** (e.g. `ATGCCA` → `ATG`, `TGC`, `GCC`, `CCA`)
- K-mers are embedded using **Word2Vec** (trained only on training sequences to prevent leakage)
- Variable-length sequences are handled via **sliding windows** (size=512, stride=256) with zero-padding

### 3. Train / Val / Test Split
A clean **stratified 3-way split** (70% / 10% / 20%) is performed at the **sequence level before windowing** — ensuring no sequence leaks across splits.

### 4. Model Training
- **CNN blocks** extract local motif patterns from the sequence
- **Bidirectional LSTM** captures long-range dependencies
- Trained with **class-weighted loss** to handle imbalanced classes
- Early stopping and ReduceLROnPlateau callbacks for stable training

### 5. Mutation Detection
After classification, the predicted sequence is aligned against the reference genome using **global pairwise alignment** (Biopython). Mutations are reported in **HGVS notation**:
- `c.45A>G` — SNP
- `c.100_101insATG` — Insertion
- `c.200_205delGCTAA` — Deletion

---

##  Getting Started

### Prerequisites
```bash
pip install -r requirements.txt
```

### Fetch Sequences from NCBI
```bash
python src/fetch_sequences.py
```

### Preprocess & Generate Embeddings
```bash
python src/preprocess.py
```

### Train the Model
```bash
python src/train.py
```

### Run the Web App
```bash
streamlit run app.py
```

---

## 🖥️ Web App

Upload any `.fasta` file and get:
- **Predicted virus variant** with sequence ID
- **Full mutation report** (SNPs, insertions, deletions) in HGVS notation
- **Downloadable TSV** mutation report

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Deep Learning | TensorFlow, Keras |
| Sequence Embedding | Gensim Word2Vec |
| Bioinformatics | Biopython, scikit-learn |
| Data Source | NCBI Entrez (via Biopython) |
| Web App | Streamlit |
| Evaluation | scikit-learn, seaborn |

---

## ⚠️ Limitations

- Mutation detection uses global pairwise alignment which may be slow for very long sequences
- Biological sequence similarity between related variants (e.g. SARS Delta vs Omicron) may affect classification confidence — CD-HIT clustering-based deduplication is identified as future work
- Model is trained on gene-specific regions (e.g. Spike for SARS-CoV-2, env for HIV) — whole-genome sequences may produce unexpected results

---

## 📄 License

MIT License — feel free to use and build on this project.

---

##  Author

**Mouli Ghosh**
[GitHub](https://github.com/mou288) • IIIT Raichur
