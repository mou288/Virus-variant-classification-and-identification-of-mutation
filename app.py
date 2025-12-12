import streamlit as st
from backend.predict_and_mutations.run import (
    classify_virus_from_text,
    load_reference_sequence,
    find_mutations,
    mutations_to_tsv_text,
)

#  PAGE CONFIG 
st.set_page_config(
    page_title="Virus Variant Classifier",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

#  HIDE STREAMLIT DEFAULT MENU / TOOLBAR 
st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
[data-testid="stToolbar"] {display: none !important;}
</style>
""", unsafe_allow_html=True)

# Force page to scroll to top on load
st.markdown("""
<script>
window.addEventListener('load', function() {
    window.parent.scrollTo(0, 0);
});
</script>
""", unsafe_allow_html=True)

#  CUSTOM CSS 
st.markdown("""
<style>
/* Global Background */
html, body, .main, [data-testid="stAppViewContainer"] {
    background: linear-gradient(to bottom, #0a0e13, #0f1419) !important;
}

.block-container {
    padding-top: 0 !important;
    padding-left: 3rem !important;
    padding-right: 3rem !important;
    max-width: 1400px !important;
}

/* Remove default gray containers */
[data-testid="stVerticalBlock"],
[data-testid="stHorizontalBlock"] {
    background: transparent !important;
}

/* Equal height columns */
[data-testid="stHorizontalBlock"] > div {
    display: flex;
    align-items: stretch;
}

[data-testid="stVerticalBlock"] > [data-testid="stVerticalBlock"] {
    height: 100%;
}

/* Hero Header */
.hero {
    width: 100%;
    background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 50%, #60a5fa 100%);
    padding: 35px 40px;
    text-align: center;
    border-radius: 16px;
    margin-bottom: 40px;
    box-shadow: 0 8px 32px rgba(59, 130, 246, 0.2);
}

.hero-title {
    font-size: 48px;
    font-weight: 800;
    margin-bottom: 8px;
    color: #ffffff;
    text-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
}

.hero-subtitle {
    font-size: 18px;
    font-weight: 400;
    color: rgba(255, 255, 255, 0.95);
    letter-spacing: 0.3px;
}

/* Section Headers */
h1, h2, h3, h4, h5, h6 {
    color: #f1f5f9 !important;
    font-weight: 600 !important;
}

/* Get Started animated text - must come after h3 to override */
.get-started-title {
    color: #60a5fa !important;
    animation: colorShift 3s ease-in-out infinite !important;
}

@keyframes colorShift {
    0%, 100% { color: #60a5fa !important; }
    50% { color: #3b82f6 !important; }
}

/* Hide anchor links next to headings */
h1 a, h2 a, h3 a, h4 a, h5 a, h6 a,
[data-testid="stHeadingWithActionElements"] button {
    display: none !important;
}

/* File Uploader */
[data-testid="stFileUploader"] {
    max-width: 600px !important;
}

[data-testid="stFileUploader"] section {
    background: transparent !important;
    border: none !important;
}

/* Target the actual drag and drop area */
[data-testid="stFileDropzone"],
[data-testid="stFileUploader"] > div > div,
[data-testid="stFileUploader"] [role="button"] {
    background: rgba(51, 65, 85, 0.5) !important;
    border: 1px solid rgba(148, 163, 184, 0.4) !important;
    border-radius: 12px !important;
    padding: 32px !important;
    transition: all 0.3s ease !important;
}

[data-testid="stFileDropzone"]:hover,
[data-testid="stFileUploader"] > div > div:hover,
[data-testid="stFileUploader"] [role="button"]:hover {
    border-color: rgba(59, 130, 246, 0.7) !important;
    background: rgba(30, 58, 138, 0.4) !important;
    transform: translateY(-2px);
}

/* Style the browse button */
[data-testid="stFileUploader"] section button {
    background: linear-gradient(135deg, #3b82f6, #2563eb) !important;
    color: white !important;
    border: none !important;
    padding: 8px 24px !important;
    font-size: 14px !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 2px 8px rgba(59, 130, 246, 0.3) !important;
    margin-top: 12px !important;
}

[data-testid="stFileUploader"] section button:hover {
    background: linear-gradient(135deg, #2563eb, #1d4ed8) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4) !important;
}

/* Target the small text and styling inside */
[data-testid="stFileUploader"] small,
[data-testid="stFileUploader"] span {
    color: #94a3b8 !important;
}

[data-testid="stFileDropzone"] label {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 8px;
}

[data-testid="stFileDropzone"] label::before {
    content: "📄";
    font-size: 48px;
    opacity: 0.6;
    display: block;
    margin-bottom: 12px;
}

[data-testid="stFileUploader"] label {
    color: #e2e8f0 !important;
    font-size: 16px !important;
    font-weight: 500 !important;
}

/* Result Cards */
.result-card {
    background: linear-gradient(to bottom, rgba(30, 41, 59, 0.5) 0%, rgba(15, 23, 42, 0.3) 50%, transparent 100%);
    padding: 30px;
    border-radius: 16px;
    margin: 24px 0;
    border: none;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    backdrop-filter: blur(10px);
}

/* Success/Info Messages */
[data-testid="stAlert"] {
    background: rgba(34, 197, 94, 0.1) !important;
    border: 1px solid rgba(34, 197, 94, 0.3) !important;
    border-radius: 10px;
    padding: 16px 20px !important;
}

[data-testid="stAlert"] [data-testid="stMarkdownContainer"] {
    color: #d1fae5 !important;
}

.stSuccess {
    background: rgba(34, 197, 94, 0.15) !important;
    border-left: 4px solid #22c55e !important;
}

.stInfo {
    background: rgba(59, 130, 246, 0.15) !important;
    border-left: 4px solid #3b82f6 !important;
}

/* DataFrames */
.dataframe, [data-testid="stDataFrame"] {
    background: rgba(15, 23, 42, 0.8) !important;
    border-radius: 12px !important;
    border: 1px solid rgba(71, 85, 105, 0.3) !important;
}

[data-testid="stDataFrame"] table {
    color: #e2e8f0 !important;
}

[data-testid="stDataFrame"] thead tr th {
    background: rgba(30, 58, 138, 0.6) !important;
    color: #f1f5f9 !important;
    font-weight: 600 !important;
    border-bottom: 2px solid rgba(59, 130, 246, 0.4) !important;
}

[data-testid="stDataFrame"] tbody tr:hover {
    background: rgba(30, 58, 138, 0.2) !important;
}

/* Download Button */
.stDownloadButton button {
    background: linear-gradient(135deg, #3b82f6, #2563eb) !important;
    color: white !important;
    border: none !important;
    padding: 12px 32px !important;
    font-size: 16px !important;
    font-weight: 600 !important;
    border-radius: 10px !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3) !important;
}

.stDownloadButton button:hover {
    background: linear-gradient(135deg, #2563eb, #1d4ed8) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(59, 130, 246, 0.4) !important;
}

/* Spinner */
.stSpinner > div {
    border-top-color: #3b82f6 !important;
}

/* Stats Badge */
.stats-badge {
    display: inline-block;
    background: linear-gradient(135deg, #3b82f6, #8b5cf6);
    color: white;
    padding: 8px 20px;
    border-radius: 20px;
    font-size: 16px;
    font-weight: 600;
    margin: 10px 0 20px 0;
    box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
}

/* Metric Styling */
[data-testid="stMetric"] {
    background: rgba(30, 41, 59, 0.4);
    padding: 20px;
    border-radius: 10px;
    border: none;
    height: 100%;
    display: flex;
    flex-direction: column;
    justify-content: center;
}

[data-testid="stMetricLabel"] {
    color: #94a3b8 !important;
    font-size: 13px !important;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

[data-testid="stMetricValue"] {
    color: #60a5fa !important;
    font-size: 22px !important;
    font-weight: 700 !important;
}

/* Text Colors */
p, li, span {
    color: #cbd5e1 !important;
}

.element-container {
    color: #e2e8f0 !important;
}

</style>
""", unsafe_allow_html=True)

#  HEADER 
st.markdown("""
<div class="hero">
    <div class="hero-title">🧬 Virus Variant Classifier</div>
    <div class="hero-subtitle">Advanced genomic analysis with HGVS mutation reporting</div>
</div>
""", unsafe_allow_html=True)

# UPLOAD SECTION 
st.markdown("### 📁 Upload FASTA Sequence")
st.markdown("<p style='color: #94a3b8; margin-bottom: 20px;'>Upload a FASTA file containing your viral sequence for classification and mutation analysis.</p>", unsafe_allow_html=True)

uploaded = st.file_uploader("", type=["fasta"], label_visibility="collapsed")

#  PROCESS FILE 
if uploaded:
    fasta_text = uploaded.read().decode()

    with st.spinner("🔬 Running classification model..."):
        seq_id, label, seq = classify_virus_from_text(fasta_text)

    # Classification Results
    st.markdown("<div class='result-card'>", unsafe_allow_html=True)
    
    st.markdown(f"### Classification Complete")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown(f"""
        <div style='background: rgba(30, 58, 138, 0.25); padding: 20px; border-radius: 10px; border-left: 4px solid #3b82f6; height: 100%;'>
            <div style='color: #94a3b8; font-size: 13px; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.5px;'>Sequence ID</div>
            <div style='color: #e2e8f0; font-size: 16px; font-family: monospace; margin-bottom: 16px;'>{seq_id}</div>
            <div style='color: #94a3b8; font-size: 13px; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.5px;'>Predicted Variant</div>
            <div style='color: #60a5fa; font-size: 22px; font-weight: 700;'>{label}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.metric("Sequence Length", f"{len(seq):,} bp")
    
    st.markdown("</div>", unsafe_allow_html=True)

    # Load reference and find mutations
    ref = load_reference_sequence(label)

    with st.spinner("🧬 Computing mutations against reference genome..."):
        muts = find_mutations(ref, seq)

    #  MUTATIONS SECTION 
    st.markdown("<div class='result-card'>", unsafe_allow_html=True)
    
    st.markdown("### 🧪 Mutation Analysis")
    
    # Mutation count badge
    st.markdown(
        f"<div class='stats-badge'>📊 {len(muts)} mutations detected</div>",
        unsafe_allow_html=True
    )
    
    if len(muts) > 0:
        st.dataframe(
            muts, 
            use_container_width=True,
            height=400
        )
        
        st.markdown("---")
        st.markdown("#### 💾 Export Results")
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            tsv = mutations_to_tsv_text(muts)
            st.download_button(
                "⬇ Download Report (TSV)",
                data=tsv,
                file_name=f"{seq_id}_{label}_mutations.tsv",
                mime="text/tab-separated-values",
            )
    else:
        st.info("No mutations detected - sequence matches reference genome.")
    
    st.markdown("</div>", unsafe_allow_html=True)

else:
    st.markdown("<div class='result-card'>", unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align: center; padding: 60px 20px; color: #94a3b8;'>
        <h3 class='get-started-title' style='font-weight: 400; font-size: 24px; margin-bottom: 16px;'>Get Started</h3>
        <p style='font-size: 15px; color: #94a3b8;'>Upload a FASTA file to begin classification and mutation analysis</p>
        <p style='font-size: 13px; margin-top: 24px; color: #64748b;'>Supported format: .fasta</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# FOOTER 
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; color: #64748b; font-size: 13px; padding: 20px;'>
    Powered by CNN and biLSTM • HGVS nomenclature compliant
</div>
""", unsafe_allow_html=True)
