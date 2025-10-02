# NLP Workflow Application

## Overview
A comprehensive Python-based NLP analysis tool built with Streamlit that allows users to upload text corpus files and perform various natural language processing tasks.

## Features
- **Named Entity Recognition (NER)**: Extract and categorize entities like persons, organizations, locations, dates, etc.
- **Sentiment Analysis**: Analyze the emotional tone and polarity of text using TextBlob
- **N-gram Analysis**: Extract and analyze common word patterns (unigrams, bigrams, trigrams, etc.)
- **Topic Modeling**: Discover latent topics using Latent Dirichlet Allocation (LDA)
- **Word Cloud Visualization**: Generate visual representations of word frequency

## Architecture
- **Frontend**: Streamlit web application
- **NLP Libraries**: 
  - spaCy (NER)
  - NLTK (text preprocessing, stopwords)
  - TextBlob (sentiment analysis)
  - scikit-learn (topic modeling, vectorization)
  - WordCloud (visualization)
- **Data Processing**: pandas, numpy

## Project Structure
- `nlp_workflow.py`: Main application file with all NLP functionality
- `sample_text.txt`: Example text file for testing
- `.gitignore`: Python-specific ignore patterns
- `pyproject.toml`: Python project configuration
- `uv.lock`: Dependency lock file

## Recent Changes (Oct 2, 2025)
- Created initial NLP workflow application with file upload support
- Implemented 5 core NLP analysis features with extensive customization options
- Added comprehensive code comments explaining how to customize each component
- Configured Streamlit workflow to run on port 5000
- Installed all required dependencies (spacy, nltk, textblob, scikit-learn, etc.)

## How to Use
1. The app runs automatically via the configured workflow
2. Upload one or more `.txt` files using the file uploader
3. Configure which analyses to perform using the sidebar checkboxes
4. Adjust parameters (n-gram size, number of topics) as needed
5. View results and export analysis report

## Customization Guide
The application includes detailed comments for customization:

1. **File Upload**: Modify `ALLOWED_EXTENSIONS` for different file types
2. **NER**: Change spaCy model in `load_spacy_model()` for different languages
3. **Sentiment**: Replace TextBlob with VADER or transformer-based models
4. **Topic Modeling**: Adjust `n_topics` parameter or switch to NMF
5. **N-grams**: Change `ngram_range` for different n-gram combinations
6. **Preprocessing**: Add lemmatization, stemming, or custom cleaning steps

## Dependencies
All managed via uv package manager:
- streamlit
- spacy (with en_core_web_sm model)
- nltk
- textblob
- scikit-learn
- wordcloud
- matplotlib
- pandas
- numpy

## User Preferences
None specified yet.
