# Sentiment Analysis on Social Media Data (US Airline Tweets)

A comprehensive sentiment analysis project using multiple machine learning and deep learning approaches on Twitter data about US airlines.

## Project Overview

This project implements and compares four different sentiment analysis models on the US Airline Twitter dataset, providing a complete end-to-end pipeline from data preprocessing to model evaluation and visualization.

### Dataset
- **Source**: US Airline Twitter Sentiment Dataset (Tweets.csv)
- **Size**: 14,640 tweets
- **Task**: 3-class sentiment classification (Negative, Neutral, Positive)
- **Features**: Tweet text, airline name, sentiment labels, negative reasons, and more

### Sentiment Distribution
- **Negative**: 62.69% (9,178 tweets)
- **Neutral**: 21.17% (3,099 tweets)
- **Positive**: 16.14% (2,363 tweets)

## Models Implemented

### 1. TF-IDF + Logistic Regression
- **Type**: Traditional ML baseline
- **Features**: TF-IDF vectors (10,000 features, unigrams + bigrams)
- **Test Accuracy**: 80.87%
- **Training Time**: ~2.9 seconds
- **Best for**: Fast inference, interpretable results

### 2. LSTM (Bidirectional)
- **Type**: Deep Learning (RNN)
- **Architecture**: Embedding → Dropout → Bidirectional LSTM → Dense layers
- **Test Accuracy**: 81.63% **BEST MODEL**
- **Training Time**: ~62 seconds
- **Best for**: Capturing sequential patterns in text

### 3. Multinomial Naive Bayes
- **Type**: Traditional ML (probabilistic)
- **Features**: CountVectorizer
- **Test Accuracy**: 78.48%
- **Training Time**: ~0.01 seconds **FASTEST**
- **Best for**: Quick baseline, resource-constrained environments

### 4. GloVe + LSTM
- **Type**: Deep Learning with pre-trained embeddings
- **Architecture**: GloVe embeddings (50d) → Bidirectional LSTM → Dense layers
- **Test Accuracy**: 81.56%
- **Training Time**: ~75 seconds
- **Best for**: Leveraging pre-trained knowledge from large corpora

## Results Summary

| Model | Accuracy | Precision (macro) | Recall (macro) | F1-Score (macro) | F1-Score (weighted) |
|-------|----------|-------------------|----------------|------------------|---------------------|
| **LSTM (Bidirectional)** | **81.63%** | **78.00%** | **73.54%** | **74.51%** | **80.94%** |
| **GloVe + LSTM** | 81.56% | 78.69% | 72.88% | 75.27% | 80.82% |
| **TF-IDF + LR** | 80.87% | 79.59% | 69.86% | 73.47% | 79.79% |
| **Naive Bayes** | 78.48% | 73.43% | 70.92% | 72.01% | 77.96% |

## Visualizations

The project includes comprehensive visualizations:
- Sentiment distribution (bar charts, pie charts)
- Tweet length analysis (histograms, violin plots, KDE)
- N-grams analysis (words, bigrams, trigrams)
- Word clouds for each sentiment class
- Confusion matrices for all models
- Training history curves (LSTM, GloVe)
- Model comparison charts
- Airline-specific sentiment analysis

## Getting Started

### Prerequisites
```bash
Python 3.7+
Jupyter Notebook
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/pvsailatha711/Sentiment-Analysis-on-Social-media-data.git
cd Sentiment-Analysis-on-Social-media-data
```

2. Install required packages:
```bash
pip install contractions emoji transformers datasets evaluate requests tqdm wordcloud seaborn plotly nltk scikit-learn tensorflow pandas numpy matplotlib
```

3. Download GloVe embeddings (optional, for GloVe model):
```bash
# GloVe 6B 50d embeddings will be downloaded automatically when running the notebook
# Or manually download from: https://nlp.stanford.edu/projects/glove/
```

### Usage

1. Ensure `Tweets.csv` is in the project directory
2. Open the Jupyter notebook:
```bash
jupyter notebook USAirline.ipynb
```
3. Run all cells to execute the complete pipeline

## Project Structure

```
Sentiment-Analysis-on-Social-media-data/
├── USAirline.ipynb          # Main notebook with complete pipeline
├── Tweets.csv               # Dataset (US Airline tweets)
├── glove.6B.50d.txt        # GloVe embeddings (downloaded automatically)
├── results/
│   ├── figures/            # All visualization outputs
│   │   ├── tweets_01_sentiment_distribution.png
│   │   ├── tweets_02_tweet_length_analysis.png
│   │   ├── tweets_03_ngrams_analysis.png
│   │   ├── tweets_04_wordclouds.png
│   │   ├── tweets_05_avg_length_by_sentiment.png
│   │   ├── tweets_06_airline_analysis.png
│   │   ├── tweets_06_metrics_comparison.png
│   │   ├── tweets_07_confusion_matrices.png
│   │   ├── tweets_08_training_time_comparison.png
│   │   ├── tweets_09_metrics_table.png
│   │   ├── tweets_lstm_training_history.png
│   │   └── tweets_glove_training_history.png
│   ├── models/             # Saved trained models
│   │   ├── tweets_tfidf_vectorizer.pkl
│   │   ├── tweets_lr_model.pkl
│   │   ├── tweets_lstm_model.h5
│   │   ├── tweets_lstm_tokenizer.pkl
│   │   ├── tweets_nb_vectorizer.pkl
│   │   ├── tweets_nb_model.pkl
│   │   ├── tweets_glove_model.h5
│   │   └── tweets_glove_embedding_matrix.pkl
│   └── tweets_model_comparison.csv  # Metrics comparison
└── README.md               # This file
```

## Methodology

### 1. Data Preprocessing
- Contraction expansion ("don't" → "do not")
- URL and mention removal
- Hashtag symbol removal (keep text)
- Emoji conversion to text descriptions
- Special character and number removal
- Lowercase conversion
- Whitespace normalization

### 2. Exploratory Data Analysis (EDA)
- Sentiment distribution analysis
- Tweet length statistics
- N-gram frequency analysis (unigrams, bigrams, trigrams)
- Word cloud generation per sentiment
- Airline-specific sentiment patterns
- Negative reason analysis

### 3. Model Training
- **Train/Val/Test Split**: 80% / 10% / 10% (stratified)
- **Training Details**:
  - Early stopping to prevent overfitting
  - Learning rate reduction on plateau
  - Dropout for regularization (LSTM models)
  - Class-weighted training for imbalanced data handling

### 4. Evaluation Metrics
- Accuracy
- Precision (macro & weighted)
- Recall (macro & weighted)
- F1-Score (macro & weighted)
- Confusion matrices
- Per-class performance analysis

## Key Findings

1. **Deep learning models** (LSTM, GloVe+LSTM) outperform traditional ML approaches
2. **Class imbalance** affects performance - negative sentiment dominates the dataset
3. **Bidirectional LSTM** captures context from both directions, leading to best performance
4. **GloVe embeddings** provide competitive results with transfer learning
5. **Naive Bayes** offers the fastest training with reasonable accuracy for quick prototyping
6. **Training time vs accuracy tradeoff**: LSTM takes longer but achieves best results

## Technologies Used

- **Languages**: Python 3.x
- **Deep Learning**: TensorFlow, Keras
- **ML Libraries**: scikit-learn
- **NLP**: NLTK, transformers
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn, plotly, wordcloud
- **Text Processing**: contractions, emoji

## Future Improvements

- Implement transformer models (BERT, RoBERTa)
- Add cross-validation for more robust evaluation
- Experiment with ensemble methods
- Try different pre-trained embeddings (FastText, Word2Vec)
- Implement attention mechanisms
- Add model interpretability (LIME, SHAP)
- Deploy as a web service (Flask/FastAPI)
- Real-time tweet sentiment analysis

## Acknowledgments

- US Airline Twitter Sentiment Dataset from Kaggle
- Stanford NLP Group for GloVe embeddings
- TensorFlow and scikit-learn communities

## Contact

For questions or feedback, please open an issue in this repository.

---

**Note**: All models have been trained and evaluated on the same dataset split to ensure fair comparison. Results may vary slightly due to random initialization in deep learning models.

