"""
NLP Workflow Application
========================
A comprehensive NLP analysis tool that processes uploaded text files and performs:
- Named Entity Recognition (NER)
- Sentiment Analysis
- N-gram Analysis
- Topic Modeling (LDA)
- Word Cloud Visualization

CUSTOMIZATION GUIDE:
-------------------
1. FILE UPLOAD: Modify `ALLOWED_EXTENSIONS` to accept different file types
2. NER: Change spaCy model in `load_spacy_model()` for different languages
3. SENTIMENT: Switch from TextBlob to VADER or other sentiment analyzers
4. TOPIC MODELING: Adjust `n_topics` parameter in LDA for more/fewer topics
5. N-GRAMS: Change `ngram_range` in `extract_ngrams()` for different n-gram sizes
6. PREPROCESSING: Modify `preprocess_text()` to add/remove cleaning steps
"""
!pip install --user spacy
import streamlit as st
import pandas as pd
import numpy as np
from collections import Counter
import re
import io

# NLP Libraries
import spacy
import nltk
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Download required NLTK data
@st.cache_resource
def download_nltk_data():
    """Download necessary NLTK datasets"""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)
    try:
        nltk.data.find('taggers/averaged_perceptron_tagger')
    except LookupError:
        nltk.download('averaged_perceptron_tagger', quiet=True)

download_nltk_data()

# CUSTOMIZATION: Change spaCy model for different languages
# Options: en_core_web_sm, en_core_web_md, en_core_web_lg (English)
#          es_core_news_sm (Spanish), de_core_news_sm (German), etc.
@st.cache_resource
def load_spacy_model(model_name="en_core_web_sm"):
    """Load spaCy model for NER"""
    try:
        return spacy.load(model_name)
    except OSError:
        st.warning(f"spaCy model '{model_name}' not found. Downloading...")
        import subprocess
        subprocess.run(['python', '-m', 'spacy', 'download', model_name])
        return spacy.load(model_name)

# CUSTOMIZATION: Add or remove text preprocessing steps
def preprocess_text(text, remove_stopwords=True, lowercase=True):
    """
    Preprocess text for NLP analysis
    
    Args:
        text: Input text string
        remove_stopwords: Whether to remove common stop words
        lowercase: Whether to convert to lowercase
    
    CUSTOMIZE: Add lemmatization, stemming, or other preprocessing steps
    """
    if lowercase:
        text = text.lower()
    
    # Remove special characters and digits (CUSTOMIZE: keep digits if needed)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    if remove_stopwords:
        from nltk.corpus import stopwords
        stop_words = set(stopwords.words('english'))
        # CUSTOMIZATION: Change 'english' to your language
        words = text.split()
        text = ' '.join([word for word in words if word not in stop_words])
    
    return text

# CUSTOMIZATION: Adjust n-gram range (1,1) for unigrams, (2,2) for bigrams, etc.
def extract_ngrams(text, n=2, top_k=20):
    """
    Extract top n-grams from text
    
    Args:
        text: Input text
        n: N-gram size (2 for bigrams, 3 for trigrams)
        top_k: Number of top n-grams to return
    
    CUSTOMIZE: Change ngram_range for different combinations
    """
    vectorizer = CountVectorizer(ngram_range=(n, n), max_features=top_k)
    try:
        X = vectorizer.fit_transform([text])
        ngrams = vectorizer.get_feature_names_out()
        counts = X.toarray()[0]
        return list(zip(ngrams, counts))
    except:
        return []

def perform_sentiment_analysis(text):
    """
    Perform sentiment analysis using TextBlob
    
    CUSTOMIZATION: Replace with VADER, transformers, or other sentiment tools
    Example with VADER:
        from nltk.sentiment import SentimentIntensityAnalyzer
        sia = SentimentIntensityAnalyzer()
        return sia.polarity_scores(text)
    """
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    
    if polarity > 0:
        sentiment = "Positive"
    elif polarity < 0:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"
    
    return {
        'sentiment': sentiment,
        'polarity': polarity,
        'subjectivity': subjectivity
    }

def extract_named_entities(text, nlp):
    """
    Extract named entities using spaCy
    
    CUSTOMIZATION: Filter specific entity types or add custom NER models
    Entity types: PERSON, ORG, GPE, DATE, MONEY, etc.
    """
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

# CUSTOMIZATION: Adjust number of topics and other LDA parameters
def perform_topic_modeling(texts, n_topics=5, n_top_words=10):
    """
    Perform topic modeling using LDA
    
    Args:
        texts: List of text documents
        n_topics: Number of topics to extract
        n_top_words: Number of top words per topic
    
    CUSTOMIZATION: 
    - Change n_topics for more/fewer topics
    - Use NMF instead of LDA: from sklearn.decomposition import NMF
    - Adjust max_features, min_df, max_df for different vocabulary sizes
    """
    # Create document-term matrix
    vectorizer = CountVectorizer(max_features=1000, min_df=2, max_df=0.8)
    try:
        doc_term_matrix = vectorizer.fit_transform(texts)
        
        # Fit LDA model
        lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
        lda.fit(doc_term_matrix)
        
        # Get feature names
        feature_names = vectorizer.get_feature_names_out()
        
        # Extract topics
        topics = []
        for topic_idx, topic in enumerate(lda.components_):
            top_words_idx = topic.argsort()[-n_top_words:][::-1]
            top_words = [feature_names[i] for i in top_words_idx]
            topics.append({
                'topic_id': topic_idx + 1,
                'words': ', '.join(top_words)
            })
        
        return topics
    except:
        return []

def generate_wordcloud(text):
    """
    Generate word cloud visualization
    
    CUSTOMIZATION: Adjust width, height, background_color, colormap, etc.
    """
    wordcloud = WordCloud(
        width=800, 
        height=400,
        background_color='white',
        colormap='viridis',  # CUSTOMIZE: try 'plasma', 'inferno', 'magma'
        max_words=100
    ).generate(text)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    return fig

# CUSTOMIZATION: Add more file types to support different formats
ALLOWED_EXTENSIONS = ['.txt', '.text']
# Example: Add .csv, .json support by parsing them differently

def read_uploaded_file(uploaded_file):
    """Read uploaded file content"""
    try:
        # CUSTOMIZATION: Add support for other file formats
        # Example for CSV: pd.read_csv(uploaded_file)['text_column']
        return uploaded_file.read().decode('utf-8')
    except:
        return uploaded_file.read().decode('latin-1')

def main():
    st.set_page_config(page_title="NLP Workflow", layout="wide")
    
    st.title("📚 NLP Analysis Workflow")
    st.markdown("""
    Upload your text corpus to perform comprehensive NLP analysis including:
    - **Named Entity Recognition (NER)**
    - **Sentiment Analysis**
    - **N-gram Extraction**
    - **Topic Modeling**
    - **Word Cloud Visualization**
    """)
    
    # Load spaCy model
    nlp = load_spacy_model()
    
    # Sidebar for configuration
    st.sidebar.header("⚙️ Analysis Configuration")
    
    # CUSTOMIZATION: Add more configuration options
    perform_ner = st.sidebar.checkbox("Named Entity Recognition", value=True)
    perform_sentiment = st.sidebar.checkbox("Sentiment Analysis", value=True)
    perform_ngrams = st.sidebar.checkbox("N-gram Analysis", value=True)
    perform_topics = st.sidebar.checkbox("Topic Modeling", value=True)
    show_wordcloud = st.sidebar.checkbox("Word Cloud", value=True)
    
    ngram_size = 2
    top_ngrams = 20
    if perform_ngrams:
        ngram_size = st.sidebar.slider("N-gram size", 1, 5, 2)
        top_ngrams = st.sidebar.slider("Top N-grams to show", 5, 50, 20)
    
    n_topics = 5
    if perform_topics:
        n_topics = st.sidebar.slider("Number of topics", 2, 10, 5)
    
    # File uploader
    st.header("📁 Upload Corpus")
    uploaded_files = st.file_uploader(
        "Upload text files",
        type=['txt', 'text'],  # CUSTOMIZATION: Add more file types
        accept_multiple_files=True
    )
    
    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} file(s) uploaded")
        
        # Read and combine all texts
        all_texts = []
        combined_text = ""
        
        for uploaded_file in uploaded_files:
            text = read_uploaded_file(uploaded_file)
            all_texts.append(text)
            combined_text += " " + text
        
        combined_text = combined_text.strip()
        
        # Display basic statistics
        st.header("📊 Corpus Statistics")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Documents", len(all_texts))
        with col2:
            st.metric("Total Words", len(combined_text.split()))
        with col3:
            st.metric("Total Characters", len(combined_text))
        with col4:
            st.metric("Avg Words/Doc", int(len(combined_text.split()) / len(all_texts)))
        
        # Preprocess text
        processed_text = preprocess_text(combined_text)
        
        # Perform analyses
        entities = []
        sentiment_results = {}
        if perform_ner:
            st.header("🏷️ Named Entity Recognition")
            with st.spinner("Extracting named entities..."):
                entities = extract_named_entities(combined_text, nlp)
                if entities:
                    # Group by entity type
                    entity_df = pd.DataFrame(entities, columns=['Entity', 'Type'])
                    entity_counts = entity_df['Type'].value_counts()
                    
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.subheader("Entity Type Distribution")
                        st.dataframe(entity_counts)
                    with col2:
                        st.subheader("All Entities")
                        st.dataframe(entity_df)
                else:
                    st.info("No named entities found")
        
        if perform_sentiment:
            st.header("😊 Sentiment Analysis")
            with st.spinner("Analyzing sentiment..."):
                sentiment_results = perform_sentiment_analysis(combined_text)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Overall Sentiment", sentiment_results['sentiment'])
                with col2:
                    st.metric("Polarity Score", f"{sentiment_results['polarity']:.3f}")
                    st.caption("Range: -1 (negative) to +1 (positive)")
                with col3:
                    st.metric("Subjectivity Score", f"{sentiment_results['subjectivity']:.3f}")
                    st.caption("Range: 0 (objective) to 1 (subjective)")
        
        if perform_ngrams:
            st.header(f"📝 {ngram_size}-gram Analysis")
            with st.spinner(f"Extracting {ngram_size}-grams..."):
                ngrams = extract_ngrams(processed_text, n=ngram_size, top_k=top_ngrams)
                if ngrams:
                    ngram_df = pd.DataFrame(ngrams, columns=[f'{ngram_size}-gram', 'Frequency'])
                    st.dataframe(ngram_df, use_container_width=True)
                else:
                    st.info(f"No {ngram_size}-grams found")
        
        if perform_topics:
            st.header("🔍 Topic Modeling (LDA)")
            with st.spinner("Discovering topics..."):
                topics = perform_topic_modeling([preprocess_text(t) for t in all_texts], 
                                               n_topics=n_topics)
                if topics:
                    st.subheader(f"Discovered {len(topics)} Topics:")
                    for topic in topics:
                        st.write(f"**Topic {topic['topic_id']}:** {topic['words']}")
                else:
                    st.info("Not enough data for topic modeling (need at least 2 documents)")
        
        if show_wordcloud:
            st.header("☁️ Word Cloud")
            with st.spinner("Generating word cloud..."):
                fig = generate_wordcloud(processed_text)
                st.pyplot(fig)
        
        # Export results option
        st.header("💾 Export Results")
        if st.button("Download Analysis Report"):
            report = f"""NLP Analysis Report
{'='*50}

Corpus Statistics:
- Documents: {len(all_texts)}
- Total Words: {len(combined_text.split())}
- Total Characters: {len(combined_text)}

"""
            if perform_sentiment:
                report += f"""
Sentiment Analysis:
- Overall Sentiment: {sentiment_results['sentiment']}
- Polarity: {sentiment_results['polarity']:.3f}
- Subjectivity: {sentiment_results['subjectivity']:.3f}

"""
            if perform_ner and entities:
                report += "\nNamed Entities:\n"
                for ent, label in entities[:20]:  # Top 20
                    report += f"- {ent} ({label})\n"
            
            st.download_button(
                label="Download Report",
                data=report,
                file_name="nlp_analysis_report.txt",
                mime="text/plain"
            )
    
    else:
        st.info("👆 Please upload one or more text files to begin analysis")
        
        # Show example
        with st.expander("ℹ️ How to use this tool"):
            st.markdown("""
            ### Quick Start Guide:
            1. **Upload Files**: Click 'Browse files' to upload one or more `.txt` files
            2. **Configure**: Use the sidebar to select which analyses to perform
            3. **View Results**: Results will appear automatically after upload
            4. **Export**: Download a summary report of your analysis
            
            ### Customization Tips:
            - See code comments for detailed customization options
            - Modify `ALLOWED_EXTENSIONS` for different file types
            - Adjust preprocessing steps in `preprocess_text()`
            - Change sentiment analyzer or topic modeling algorithm
            - Add custom entity types for domain-specific NER
            """)

if __name__ == "__main__":
    main()
