#  MSDS 453 Final Project: Topic Discovery in Cardiovascular Clinical Notes

This repository contains my final project for MSDS 453 (AI and Natural Language Processing), comparing text representation methods for discovering clinical topics in cardiovascular discharge summaries and establishing a theoretical framework connecting topic modeling to Retrieval-Augmented Generation (RAG) system design.

## 📌 Project Overview

I'm analyzing 10,000 cardiovascular discharge summaries from the MIMIC-III clinical database to compare TF-IDF and Doc2Vec representations for unsupervised topic discovery. The broader goal is to demonstrate that topic modeling quality metrics can predict RAG retrieval performance without building the full system, since both rely fundamentally on cosine similarity.

## 🔬 Key Findings

**K-Means Clustering Reveals Multi-Topic Document Structure:**

- Silhouette scores of 0.024 (TF-IDF, k=7) and 0.003 (Doc2Vec) indicate no meaningful hard cluster structure
- Multiple clusters dominated by high-frequency administrative and medication terms (tablet, pacemaker, refills, disp, release) spanning clinical categories, limiting interpretability
- Root cause: Discharge summaries are inherently multi-topic (primary diagnosis + comorbidities + procedures + documentation boilerplate)

**LDA Topic Modeling Proves More Appropriate:**

- Coherence score of 0.6251 at 25 topics: the optimal configuration, with coherence increasing monotonically from 0.4037 at 5 topics
- Discovered clinically meaningful themes: acute MI (coronary, catheterization, stent, LAD, RCA), valvular heart disease (mitral, regurgitation, echocardiogram), arrhythmia/device management (pacemaker, defibrillator, tachycardia), CABG (graft, bypass, saphenous), and anticoagulation therapy (warfarin, amiodarone, heparin)
- Allows documents to be mixtures of topics, matching clinical reality

**Supervised Classification Validates Topic Quality:**

- Logistic Regression achieved 97% accuracy using K-Means cluster labels (TF-IDF)
- 94% accuracy across 25 LDA topic classes (Doc2Vec)
- Demonstrates discovered categories are learnable and meaningful

## 🎯 Methodological Contributions

1. **Multi-Topic Document Discovery:** Demonstrated that comprehensive clinical documents require probabilistic topic modeling rather than hard partitioning. K-Means failure (silhouette <0.10) validates LDA necessity.

2. **Topic Modeling → RAG Framework:** Established theoretical connection showing that LDA topic distributions serve as probabilistic semantic indices for RAG retrieval. Topic coherence predicts retrieval quality before system implementation.

3. **Medical Term Normalization:** Identified critical preprocessing challenges where medical synonyms fragment (MI/AMI/STEMI/NSTEMI treated separately) and developed consolidation strategies using a 200+ term normalization dictionary.

4. **Documentation Artifact Identification:** Isolated boilerplate language as a dominant latent topic (Topic 3, dominant in 1,675 documents), informing future preprocessing strategies.

## 📊 Technical Approach

**Data:**

- MIMIC-III NOTEEVENTS: 10,000 cardiovascular discharge summaries
- Filtered for discharge summary document type (consistent structure)
- Cardiology-specific content (MI, heart failure, arrhythmias, CAD)
- Access compliant with MIMIC-III data use agreement

**Methods:**

- TF-IDF vectorization (3,000 features, max_df = 0.5, unigrams)
- Doc2Vec embeddings (100 dimensions, PV-DM architecture)
- K-Means clustering with silhouette-based k optimization (tested k = 3, 5, 7, 10, 12, 15)
- LDA topic modeling (tested 5–25 topics, selected k = 25 via C_v coherence; CountVectorizer max_features = 3,000, max_df = 0.7, min_df = 10)
- Supervised classification (Logistic Regression, Linear SVM)

**Evaluation:**

- Silhouette coefficient (cosine distance) for K-Means quality
- C_v coherence and perplexity for LDA topic interpretability
- Classification accuracy and F1 for topic learnability
- RAG quality metrics (doc-to-centroid similarity, coverage at thresholds, representative document selection)

## 🛠 Tools & Technologies

- Python 3.12
- NLP Libraries: NLTK, Gensim, scikit-learn
- Analysis: pandas, NumPy, Matplotlib, Seaborn
- Clinical NLP: Custom medical abbreviation normalization dictionary (200+ terms)
- Documentation: Jupyter Notebooks, CMOS-formatted report

## 📂 Repository Structure

```
MSDS453-Final-Project/
│
├── scripts/
│   └── clinical_note_clustering.ipynb       # Main analysis notebook
│
├── data/
  │   └── NOTEEVENTS.csv                     # Text dataset
|
├── report/
│   └── Pasaye_Project-Report-Final.docx     # Final report (CMOS format)
│
├── visualizations/
│   ├── cluster_comparison.png               # TF-IDF vs Doc2Vec PCA plots (Figure 1)
│   ├── lda_topics.png                       # LDA topic distribution (Figure 2)
│   └── rag_design_analysis.png              # RAG quality metrics (Figure 3)
│
├── .gitignore
|
└── README.md
```

## 🚀 Key Results

**Quantitative Metrics:**

- LDA Coherence: 0.6251 (25 topics)
- LDA Perplexity: 1,133.64 (25 topics)
- K-Means Silhouette: 0.024 (TF-IDF, k=7) / 0.003 (Doc2Vec)
- Supervised Classification: 94–97% accuracy
- RAG Coverage (>0.3 similarity): 96.2%; (>0.5 similarity): 27.7%

**Qualitative Findings:**

- LDA discovers clinically interpretable cardiovascular themes
- TF-IDF produces more coherent topics than Doc2Vec for clinical text
- Multi-label probabilistic representations are essential for comprehensive clinical documentation
- Topic modeling provides principled pathway from exploratory analysis to RAG system design

## 💡 Information Extraction Challenges Addressed

**Medical Synonym Fragmentation:** Myocardial infarction appears as MI/AMI/STEMI/NSTEMI/heart attack — five separate weak signals. Solution: 200+ term normalization dictionary consolidating medical synonyms.

**Clinical Boilerplate Dominance:** Generic documentation phrases and medication refill text dominated term frequency. Solution: max_df filtering and clinical-specific stopwords. LDA's dedicated boilerplate topic (Topic 3) further isolates this artifact.

**Lost Phenotype Information:** Numeric values stripped during preprocessing, losing ejection fraction distinctions between HFrEF and HFpEF. Solution: Preserve quantitative clinical markers in future iterations.

## 📈 Future Work

**Near-term:**

- Explore topic counts beyond 25 to assess further coherence gains
- Develop a cardiovascular-domain-specific stopword list to suppress boilerplate topics
- Incorporate clinical expert review to validate LDA topic assignments

**Long-term:**

- Full RAG system implementation and benchmark evaluation against clinical QA datasets
- Fine-tune BioBERT on cardiovascular discharge summaries
- UMLS concept-based vectors for automatic medical synonym resolution
- Extension to multiple institutions and non-cardiovascular clinical domains

## 📝 Academic Integrity

This project analyzes the MIMIC-III Critical Care Database, a publicly available de-identified clinical dataset. All work adheres to Northwestern University's academic integrity policies and HIPAA compliance requirements for clinical data analysis.

## 📧 Contact

**Joshua Pasaye**  
Northwestern University MSDS  
GitHub: [jep9731](https://github.com/jep9731/)

---

*This project demonstrates applied NLP skills in healthcare informatics, unsupervised learning, topic modeling, and the design of intelligent information retrieval systems.*
