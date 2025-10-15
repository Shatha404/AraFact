
## Arabic Fact-Checking with AraBERT
An NLP project for automating Arabic fact-checking using the AraFacts dataset and transformer-based models.
### Project Overview
This project implements an automated fact-checking system for Arabic claims using natural language processing techniques. We leveraged the AraFacts dataset and fine-tuned pre-trained Arabic BERT models to classify claims into multiple categories.
### Dataset
AraFacts is a specialized dataset for Arabic fact-checking containing:
- 6,000+ naturally occurring claims
- Professional verification and labeling
- Five categories: False, Partly-False, True, Sarcasm, and Unverifiable

### Methodology
#### Data Cleaning
- Missing & Duplicate Data: Removed incomplete and duplicate records
- Normalization: Standardized text to UTF-8 Unicode format
- Diacritic Removal: Removed diacritics to standardize word forms
- Stopword Filtering: Filtered common Arabic stopwords (و، في، على)
- Stemming: Applied Farasa for prefix/suffix removal and stemming

#### Preprocessing
1- Tokenization: Converted text into tokens using AraBERT tokenizer.
2- Padding & Truncation: Standardized input lengths for batch processing.
3- Label Encoding: Converted categorical labels to numerical values.
4- Dataset Format: Converted to Hugging Face Dataset format.

#### Model Training
- Primary Model: bert-base-arabertv02 (AraBERT)
- Train/Test Split: 80/20
- Optimizer: AdamW
- Batch Size: 16
- Epochs: 5

#### Performance:
- Accuracy improved from 69% → 91%
- Strong performance for False and Partly-False classes
- Weaker performance for True, Sarcasm, and Unverifiable

### Model Comparison:

| Model          | Accuracy              | Notes            |
| -------------- | --------------------- | ---------------- |
| **AraBERT**    | ~91%                  | Best performance |
| **MARBERT**    | Similar to AraBERT    |                  |
| **DistilBERT** | <70%                  | Poor performance |
| **CAMeLBERT**  | Similar to DistilBERT |                  |

#### AraBERT Classification Results
| **Class**        | **Precision** | **Recall** | **F1-Score** |
| ---------------- | ------------- | ---------- | ------------ |
| **False**        | 0.89          | 0.96       | 0.93         |
| **Partly-False** | 0.92          | 0.96       | 0.94         |
| **True**         | 0.78          | 0.04       | 0.08         |
| **Sarcasm**      | 0.00          | 0.00       | 0.00         |
| **Unverifiable** | 0.00          | 0.00       | 0.00         |

Overall Accuracy: 91%

### Challenges & Solutions
- Class Imbalance → Applied class weights and oversampling
- Arabic Language Complexity → Used AraBERT tokenizer to handle diacritics and morphology
- Overfitting → Suggested dropout, weight decay, and early stopping

### Conclusion
An AraBERT-based Arabic fact-checking classifier was developed using the AraFacts dataset, achieving an impressive 91% accuracy.
Despite challenges such as class imbalance and Arabic linguistic complexity, the model demonstrates strong potential for real-world Arabic NLP applications.

### Tools & Libraries
- Python
- Hugging Face Transformers
- Pandas
- Matplotlib, 
- Farasa Arabic NLP Toolkit
