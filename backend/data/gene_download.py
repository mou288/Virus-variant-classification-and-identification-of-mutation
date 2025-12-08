import os
import random
from io import StringIO
from Bio import Entrez, SeqIO
from dotenv import load_dotenv

load_dotenv()

Entrez.email = os.getenv("NCBI_EMAIL")


MAX_RECORDS_PER_VIRUS = 5000
TARGET_SEQUENCES_PER_VARIANT = 1000
MAX_N_PERCENTAGE = 3.0

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw")


os.makedirs(OUTPUT_DIR, exist_ok=True)


VIRUS_CONFIG = {

    # ---------------- HIV-1 ----------------
    "Human immunodeficiency virus 1": {
        "gene_name": "env",
        "min_length": 2300,
        "max_length": 2700,
       
        "variants": [
            {
                "search_term":
                    "Clade B OR subtype B OR B1 OR B-1",
                "output_file": "hiv_clade_B_env.fasta"
            },
            {
                "search_term":
                    "Clade C OR subtype C OR C1 OR C2 OR C-1 OR C-2",
                "output_file": "hiv_clade_C_env.fasta"
            }
        ]
    },

    # ---------------- SARS-CoV-2 ----------------
    "SARS-CoV-2": {
        "gene_name": "S",
        "min_length": 3500,
        "max_length": 4000,
        
        "variants": [
            {
                "search_term":
                    "Omicron OR BA.1 OR BA.2 OR BA.3 OR BA.4 OR BA.5 OR XBB OR B.1.1.529",
                "output_file": "sars_omicron_S.fasta"
            },
            {
                "search_term":
                    "Delta OR B.1.617.2 OR AY. OR AY.1 OR AY.2 OR AY.3",
                "output_file": "sars_delta_S.fasta"
            }
        ]
    },

    # ---------------- Dengue ----------------
    "Dengue virus": {
        "gene_name": "E",
        "min_length": 1300,
        "max_length": 1700,
      
        "variants": [
            
            {
                "search_term":
                    "Genotype IV OR genotype 4 OR IV OR DENV-1 IV OR DENV1-IV",
                "output_file": "denv1_genotype_IV_E.fasta"
            },
            {
                "search_term":
                    "Genotype V OR genotype 5 OR V",
                "output_file": "denv1_genotype_V_E.fasta"
            }
        ]
    },

    # ---------------- Hepatitis B ----------------
    "Hepatitis B virus": {
        "gene_name": "S",
        "min_length": 600,
        "max_length": 1300,
     
        "variants": [
           
            {
                "search_term":
                    "Genotype C OR C1 OR C2 OR subgenotype C1 OR subgenotype C2",
                "output_file": "hbv_genotype_C_S.fasta"
            },
            {
                "search_term":
                    "Genotype D OR D1 OR D2 OR subgenotype D1 OR subgenotype D2",
                "output_file": "hbv_genotype_D_S.fasta"
            }
        ]
    },

    # ---------------- Hepatitis C ----------------
    "Hepatitis C virus": {
        "gene_name": "E2",
        "min_length": 900,
        "max_length": 1200,
       
        "variants": [
            {
                "search_term":
                    "Genotype 1a OR 1a OR subtype 1a",
                "output_file": "hcv_genotype_1a_E2.fasta"
            },
            {
                "search_term":
                    "Genotype 3 OR genotype 3a OR genotype 3b OR 3a OR 3b",
                "output_file": "hcv_genotype_3_E2.fasta"
            }
        ]
    },

    # ---------------- Measles ----------------
    "Measles morbillivirus": {
        "gene_name": "H",
        "min_length": 1700,
        "max_length": 2100,
       
        "variants": [
            {
                "search_term":
                    "Genotype B3 OR lineage B3 OR B3",
                "output_file": "measles_genotype_B3_H.fasta"
            },
            {
                "search_term":
                    "Genotype D8 OR lineage D8 OR D8",
                "output_file": "measles_genotype_D8_H.fasta"
            },
            {
                "search_term":
                    "Genotype H1 OR lineage H1 OR H1",
                "output_file": "measles_genotype_H1_H.fasta"
            }
        ]
    },

    # ---------------- Influenza A ----------------
    "Influenza A virus": {
        "gene_name": "HA",
        "min_length": 1650,
        "max_length": 1850,
     
        "variants": [
            {
                "search_term":
                    "H1N1 OR A/H1N1 OR H1 OR swine flu",
                "output_file": "influenza_A_H1N1_HA.fasta"
            },
            {
                "search_term":
                    "H3N2 OR A/H3N2 OR H3",
                "output_file": "influenza_A_H3N2_HA.fasta"
            },
            {
                "search_term":
                    "H5N1 OR A/H5N1 OR H5 OR avian influenza",
                "output_file": "influenza_A_H5N1_HA.fasta"
            }
        ]
    }
}


def sanitize_filename(s: str):
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in s)


# ================================
# MAIN LOOP
# ================================
for virus, cfg in VIRUS_CONFIG.items():
    print("\n========== PROCESSING:", virus, "==========")

    min_len, max_len = cfg["min_length"], cfg["max_length"]

    # Build gene query
    gene_field = cfg["gene_name"]

    if gene_field is None:
        gene_query = None
    elif isinstance(gene_field, list):
        gene_query = " OR ".join([f'"{g}"[All Fields]' for g in gene_field])
    else:
        gene_query = f'"{gene_field}"[All Fields]'

    for variant in cfg["variants"]:
        term = variant["search_term"]
        outfile = os.path.join(OUTPUT_DIR, sanitize_filename(variant["output_file"]))

        if gene_query:
            query = (
                f'"{virus}"[Organism] AND ({gene_query}) AND ({term}) AND '
                f'{min_len}:{max_len}[SLEN]'
            )
        else:
            query = (
                f'"{virus}"[Organism] AND ({term}) AND {min_len}:{max_len}[SLEN]'
            )

        print("\nVariant:", term)
        print("Query:", query)

        try:
            h = Entrez.esearch(db="nucleotide", term=query, retmax=MAX_RECORDS_PER_VIRUS)
            rec = Entrez.read(h); h.close()
            ids = rec["IdList"]

            if not ids:
                print("No records found.")
                continue

            print(f"Found {len(ids)} sequences.")

            h = Entrez.efetch(db="nucleotide", id=ids, rettype="fasta", retmode="text")
            fasta_text = h.read(); h.close()

            parsed = []

            for r in SeqIO.parse(StringIO(fasta_text), "fasta"):
                if not (min_len <= len(r.seq) <= max_len):
                    continue
                if (r.seq.upper().count("N") / len(r.seq)) * 100 > MAX_N_PERCENTAGE:
                    continue

                r.description += f" | variant={term}"
                parsed.append(r)

            if len(parsed) > TARGET_SEQUENCES_PER_VARIANT:
                parsed = random.sample(parsed, TARGET_SEQUENCES_PER_VARIANT)

            SeqIO.write(parsed, outfile, "fasta")
            print(f"Saved {len(parsed)} → {outfile}")

        except Exception as e:
            print("ERROR:", e)

print("\n======= ALL DOWNLOADS COMPLETE =======\n")
