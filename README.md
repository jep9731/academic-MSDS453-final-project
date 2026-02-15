# MSDS 453 Final Project: Topic Discovery in Cardiovascular Clinical Notes

This repository contains my final project for MSDS 453 (AI and Natural Language Processing), comparing text representation methods for discovering clinical topics in cardiovascular discharge summaries and establishing a theoretical framework connecting topic modeling to Retrieval-Augmented Generation (RAG) system design.

## 📌 Project Overview

I'm analyzing 10,000 cardiovascular discharge summaries from the MIMIC-III clinical database to compare TF-IDF and Doc2Vec representations for unsupervised topic discovery. The broader goal is to demonstrate that topic modeling quality metrics can predict RAG retrieval performance without building the full system, since both rely fundamentally on cosine similarity.

## 🔬 Key Findings

**K-Means Clustering Reveals Multi-Topic Document Structure:**
- Silhouette scores of 0.025 (TF-IDF) and 0.021 (Doc2Vec) indicate no meaningful hard cluster structure
- 62.7% of documents collapsed into a single "amoeba cluster" with generic terms like "patient", "hospital", "admission"
- Root cause: Discharge summaries are inherently multi-topic (35% primary diagnosis + 25% comorbidities + 20% procedures + 20% other)

**LDA Topic Modeling Proves More Appropriate:**
- Coherence score of 0.46 at 15 topics = substantially better than hard clustering
- Discovered clinically meaningful themes: acute MI, heart failure, arrhythmias, procedures, diabetes management
- Allows documents to be mixtures of topics, matching clinical reality

**Supervised Classification Validates Topic Quality:**
- Logistic Regression achieved 99% accuracy using K-Means cluster labels (TF-IDF)
- 96% accuracy across 15 LDA topic classes (Doc2Vec)
- Demonstrates discovered categories are learnable and meaningful

## 🎯 Methodological Contributions

**1. Multi-Topic Document Discovery:**
Demonstrated that comprehensive clinical documents require probabilistic topic modeling rather than hard partitioning. K-Means failure (silhouette <0.10) validates LDA necessity.

**2. Topic Modeling → RAG Framework:**
Established theoretical connection showing that LDA topic distributions serve as probabilistic semantic indices for RAG retrieval. Topic coherence predicts retrieval quality before system implementation.

**3. Medical Term Normalization:**
Identified critical preprocessing challenges where medical synonyms fragment (MI/AMI/STEMI/NSTEMI treated separately) and developed consolidation strategies improving coherence from 0.30 to 0.46.

## 📊 Technical Approach

**Data:**
- MIMIC-III NOTEEVENTS: 10,000 cardiovascular discharge summaries
- Filtered for discharge summary document type (consistent structure)
- Cardiology-specific content (MI, heart failure, arrhythmias, CAD)

**Methods:**
- TF-IDF vectorization (3,000 features, unigrams, optimized filtering)
- Doc2Vec embeddings (100 dimensions, PV-DM architecture)
- K-Means clustering (automatic k optimization via silhouette scores)
- LDA topic modeling (tested 5-25 topics, selected k=15 via C_v coherence)
- Supervised classification (Logistic Regression, SVM)

**Evaluation:**
- Silhouette coefficient for cluster quality
- C_v coherence for topic interpretability
- Classification accuracy for topic learnability
- RAG quality metrics (doc-to-centroid similarity, coverage)

## 🛠 Tools & Technologies

- **Python 3.12**
- **NLP Libraries:** NLTK, gensim, scikit-learn
- **Analysis:** pandas, numpy, matplotlib, seaborn
- **Clinical NLP:** Custom medical abbreviation normalization
- **Documentation:** Jupyter notebooks, comprehensive reports

## 📂 Repository Structure
```
MSDS453-Final-Project/
│
├── scripts/
│   ├── clinical_note_clustering.ipynb       # Main analysis script
│   └── rag_analysis_module.py               # RAG design framework
│
├── report/
│   └── Interim_Report_A1_FINAL_CLEAN.docx   # Final report (CMOS format)
│
├── visualizations/
│   ├── cluster_comparison.png               # TF-IDF vs Doc2Vec PCA plots
│   ├── lda_topics.png                       # LDA topic distribution
│   └── rag_design_analysis.png              # RAG quality metrics
│
└── README.md                                
```

## 🚀 Key Results

**Quantitative Metrics:**
- LDA Coherence: 0.46 (15 topics)
- K-Means Silhouette: 0.025 (3 clusters)
- Supervised Classification: 96-99% accuracy
- RAG Coverage (>0.5 similarity): 75-85% (projected from topic coherence)

**Qualitative Findings:**
- LDA discovers clinically interpretable cardiovascular themes
- TF-IDF produces more coherent topics than Doc2Vec for clinical text
- Multi-label probabilistic representations are essential for comprehensive clinical documentation
- Topic modeling provides principled pathway from exploratory analysis to RAG system design

## 💡 Information Extraction Challenges Addressed

**Medical Synonym Fragmentation:**
Myocardial infarction appears as MI/AMI/STEMI/NSTEMI/heart attack—five separate weak signals. Solution: 200+ term normalization dictionary consolidating medical synonyms.

**Clinical Boilerplate Dominance:**
Generic documentation phrases ("patient presented with", "hospital course") dominated term frequency. Solution: Aggressive filtering (max_df=0.5) and clinical-specific stop-words.

**Lost Phenotype Information:**
Numeric values stripped during preprocessing, losing ejection fraction distinctions between HFrEF and HFpEF. Solution: Preserve quantitative clinical markers in future iterations.

## 📈 Future Work

**Short-term:**
- Implement comprehensive medical term normalization (200+ mappings)
- Aggressive clinical boilerplate filtering
- Medical-aware stemming

**Medium-term:**
- UMLS concept-based vectors for automatic synonym resolution
- Supervised topic seeding with clinical categories
- Hierarchical topic modeling for multi-granularity structure

**Long-term:**
- Fine-tune BioBERT on cardiovascular discharge summaries
- Implement full RAG system using discovered topic structure
- Clinical expert validation of topic assignments

## 📝 Academic Integrity

This project analyzes the MIMIC-III Critical Care Database, a publicly available de-identified clinical dataset. All work adheres to Northwestern University's academic integrity policies and HIPAA compliance requirements for clinical data analysis.

## 📧 Contact

**Joshua Pasaye**  
Northwestern University MSDS  
GitHub: [jep9731](https://github.com/jep9731/)

---

*This project demonstrates applied NLP skills in healthcare informatics, unsupervised learning, topic modeling, and the design of intelligent information retrieval systems.*
