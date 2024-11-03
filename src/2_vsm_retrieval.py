import os
import re
import csv
import math
import logging
import pandas as pd
from collections import defaultdict
from nltk.corpus import stopwords, wordnet, words as nltk_words
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from typing import List, Tuple, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("vsm_retrieval.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize global variables
stop_words = set(stopwords.words('english'))  # Load English stop words
lemmatizer = WordNetLemmatizer()  # Initialize a lemmatizer for word normalization
english_words = set(nltk_words.words())  # Create a set of English words for quick lookup

def clean_text(text: str) -> str:
    """
    Cleans the input text by removing numbers, punctuation, single-character words, and extra spaces.

    Args:
        text (str): The raw text to clean.

    Returns:
        str: Cleaned text string.
    """
    logger.debug("Starting text cleaning.")
    text = re.sub(r'\d+', ' ', text)  # Remove numbers
    text = re.sub(r'[^\w\s]', ' ', text)  # Remove punctuation
    text = re.sub(r'\b\w\b', ' ', text)  # Remove single-character words
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    logger.debug("Text cleaning completed.")
    return text

def tokenize_text(text: str) -> List[str]:
    """
    Tokenizes the input text into individual words.

    Args:
        text (str): The text to tokenize.

    Returns:
        List[str]: List of word tokens.
    """
    logger.debug("Starting text tokenization.")
    tokens = word_tokenize(text)
    logger.debug(f"Tokens after tokenization: {tokens}")
    return tokens

def normalize_tokens(tokens: List[str]) -> List[str]:
    """
    Converts all tokens to lowercase.

    Args:
        tokens (List[str]): List of tokens to normalize.

    Returns:
        List[str]: List of lowercase tokens.
    """
    logger.debug("Starting token normalization to lowercase.")
    normalized = [token.lower() for token in tokens]
    logger.debug(f"Tokens after normalization: {normalized}")
    return normalized

def remove_stopwords_from_tokens(tokens: List[str]) -> List[str]:
    """
    Removes English stopwords from the list of tokens.

    Args:
        tokens (List[str]): List of tokens to filter.

    Returns:
        List[str]: List of tokens without stopwords.
    """
    logger.debug("Starting stopword removal.")
    filtered = [word for word in tokens if word not in stop_words]
    logger.debug(f"Tokens after stopword removal: {filtered}")
    return filtered

def get_wordnet_pos(tag: str) -> str:
    """
    Maps NLTK POS tags to WordNet POS tags for lemmatization.

    Args:
        tag (str): The POS tag from NLTK.

    Returns:
        str: Corresponding POS tag for WordNet.
    """
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def lemmatize_tokens(tokens: List[str]) -> List[str]:
    """
    Lemmatizes the tokens using their POS tags for more accurate lemmatization.

    Args:
        tokens (List[str]): List of tokens to lemmatize.

    Returns:
        List[str]: List of lemmatized words.
    """
    logger.debug("Starting lemmatization of tokens.")
    pos_tags = pos_tag(tokens)
    lemmatized = [
        lemmatizer.lemmatize(word, get_wordnet_pos(tag))
        for word, tag in pos_tags
        if word.isalpha() and len(word) > 1
    ]
    logger.debug(f"Tokens after lemmatization: {lemmatized}")
    return lemmatized

def remove_non_english_words_from_tokens(tokens: List[str]) -> List[str]:
    """
    Removes any words that are not in the English dictionary.

    Args:
        tokens (List[str]): List of word tokens.

    Returns:
        List[str]: List of English words.
    """
    logger.debug("Starting removal of non-English words.")
    filtered = [word for word in tokens if word in english_words]
    logger.debug(f"Tokens after removing non-English words: {filtered}")
    return filtered

def process_text(text: str) -> List[str]:
    """
    Combines all text preprocessing steps: cleaning, tokenizing, normalizing, removing stopwords, lemmatizing,
    and removing non-English words.

    Args:
        text (str): The raw text to process.

    Returns:
        List[str]: List of processed tokens.
    """
    logger.info("Starting text processing.")
    text = clean_text(text)
    tokens = tokenize_text(text)
    tokens = normalize_tokens(tokens)
    tokens = remove_stopwords_from_tokens(tokens)
    tokens = lemmatize_tokens(tokens)
    #tokens = remove_non_english_words_from_tokens(tokens)
    logger.info("Text processing completed.")
    return tokens

def read_inverted_index(inverted_index_csv: str) -> Tuple[Dict[str, Dict[str, float]], int]:
    """
    Reads the inverted index from a CSV file and returns a dictionary.

    Args:
        inverted_index_csv (str): Path to the inverted index CSV file.

    Returns:
        Tuple[Dict[str, Dict[str, float]], int]:
            - Dictionary with words as keys and dictionaries of documents and tf as values.
            - Total number of unique documents.
    """
    logger.info(f"Reading inverted index from {inverted_index_csv}.")
    inverted_index = defaultdict(dict)
    total_documents = set()

    try:
        df = pd.read_csv(inverted_index_csv)
    except Exception as e:
        logger.error(f"Error reading inverted index file {inverted_index_csv}: {e}")
        return inverted_index, 0

    for idx, row in df.iterrows():
        word = row['Word']
        docs = row['Documents'].split(', ')
        for doc_tf in docs:
            try:
                doc_name, tf = doc_tf.split(':')
                tf = float(tf)
                inverted_index[word][doc_name] = tf
                total_documents.add(doc_name)
            except ValueError as ve:
                logger.warning(f"Error splitting doc_tf '{doc_tf}' into doc_name and tf: {ve}")

    total_documents_count = len(total_documents)
    logger.info(f"Total documents in inverted index: {total_documents_count}")
    return inverted_index, total_documents_count

def compute_idf(inverted_index: Dict[str, Dict[str, float]], total_documents: int) -> Dict[str, float]:
    """
    Computes the Inverse Document Frequency (IDF) for each term in the inverted index.

    Args:
        inverted_index (Dict[str, Dict[str, float]]): The inverted index.
        total_documents (int): Total number of documents.

    Returns:
        Dict[str, float]: Dictionary containing IDF values for each term.
    """
    logger.info("Computing IDF for each term in the inverted index.")
    idf = {}
    for term, docs in inverted_index.items():
        df = len(docs)
        idf_value = math.log((total_documents / (1 + df)))  # Adding 1 to avoid division by zero
        idf[term] = idf_value
        logger.debug(f"Term: {term}, DF: {df}, IDF: {idf_value:.4f}")
    logger.info("IDF computation completed.")
    return idf

def retrieve_documents(
    query: str,
    inverted_index: Dict[str, Dict[str, float]],
    idf: Dict[str, float],
    total_documents: int
) -> List[Tuple[str, float]]:
    """
    Retrieves and ranks relevant documents for a query using the Vector Space Model.

    Args:
        query (str): The query text.
        inverted_index (Dict[str, Dict[str, float]]): The inverted index.
        idf (Dict[str, float]): IDF values for each term.
        total_documents (int): Total number of documents.

    Returns:
        List[Tuple[str, float]]: List of tuples containing (document_name, similarity_score).
    """
    logger.info("Processing query.")
    query_tokens = process_text(query)
    logger.info(f"Processed query tokens: {query_tokens}")

    # Calculate tf-idf weights for the query
    query_tf = defaultdict(int)
    for term in query_tokens:
        query_tf[term] += 1
    query_tf_idf = {}
    for term, tf in query_tf.items():
        if term in idf:
            query_tf_idf[term] = (tf / len(query_tokens)) * idf[term]
        else:
            query_tf_idf[term] = 0.0  # Term not in inverted index, IDF is zero
    logger.debug(f"Query TF-IDF weights: {query_tf_idf}")

    # Initialize scores for documents
    scores = defaultdict(float)
    for term, q_weight in query_tf_idf.items():
        if term in inverted_index:
            for doc, doc_tf in inverted_index[term].items():
                doc_weight = doc_tf * idf[term]
                scores[doc] += q_weight * doc_weight
                logger.debug(f"Updating score for document {doc}: {scores[doc]:.6f}")

    # Calculate norms for documents
    doc_norms = defaultdict(float)
    for term, docs in inverted_index.items():
        for doc, doc_tf in docs.items():
            weight = doc_tf * idf[term]
            doc_norms[doc] += weight ** 2
    for doc in doc_norms:
        doc_norms[doc] = math.sqrt(doc_norms[doc])
        logger.debug(f"Document norm for {doc}: {doc_norms[doc]:.6f}")

    # Calculate norm for the query
    query_norm = math.sqrt(sum([w ** 2 for w in query_tf_idf.values()]))
    logger.debug(f"Query vector norm: {query_norm:.6f}")

    # Calculate final similarity scores (cosine similarity)
    for doc in scores:
        if doc_norms[doc] != 0 and query_norm != 0:
            scores[doc] = scores[doc] / (doc_norms[doc] * query_norm)
        else:
            scores[doc] = 0.0
        logger.debug(f"Final similarity score for document {doc}: {scores[doc]:.6f}")

    # Rank documents based on similarity scores
    ranked_docs = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    top_10_docs = ranked_docs[:10]
    logger.info(f"Retrieved top {len(top_10_docs)} documents for the query.")

    return top_10_docs

def read_queries(queries_csv: str) -> List[Tuple[Any, str]]:
    """
    Reads queries from a CSV file and returns a list of tuples containing (query_id, query_text).

    Args:
        queries_csv (str): Path to the queries CSV file.

    Returns:
        List[Tuple[Any, str]]: List of tuples containing (query_id, query_text).
    """
    logger.info(f"Reading queries from {queries_csv}.")
    queries = []
    try:
        df = pd.read_csv(queries_csv, sep='\t')
    except Exception as e:
        logger.error(f"Error reading queries file {queries_csv}: {e}")
        return queries

    for idx, row in df.iterrows():
        query_id = row['Query number']
        query_text = row['Query']
        queries.append((query_id, query_text))

    logger.info(f"Total queries read: {len(queries)}.")
    return queries

def save_results_to_csv(results: List[Tuple[Any, str]], output_csv: str):
    """
    Saves query results to a CSV file.

    Args:
        results (List[Tuple[Any, str]]): List of tuples containing (query_id, document_title).
        output_csv (str): Path to the CSV file to save results.
    """
    logger.info(f"Saving results to {output_csv}.")
    try:
        with open(output_csv, 'w', newline='', encoding='utf-8') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['Query_number', 'doc_number'])
            for query_id, doc_title in results:
                writer.writerow([query_id, doc_title])
        logger.info(f"Results successfully saved to {output_csv}.")
    except Exception as e:
        logger.error(f"Error saving results to {output_csv}: {e}")

def main():
    """
    Main function to execute the script.
    """
    logger.info("Starting document retrieval using Vector Space Model.")

    # Path to the merged inverted index CSV file
    inverted_index_csv = '../merged_inverted_index_pandas_big_ranked_non_en.csv'

    # Path to the queries CSV file
    queries_csv = '../data/query/dev_queries.tsv'  # Ensure this file contains columns: 'Query number' and 'Query'

    # Path to the output results CSV file
    output_csv = '../results/result.csv'

    # Read the inverted index and compute IDF
    inverted_index, total_documents = read_inverted_index(inverted_index_csv)
    if total_documents == 0:
        logger.error("No documents found in the inverted index. Exiting the process.")
        return
    idf = compute_idf(inverted_index, total_documents)

    # Read the queries
    queries = read_queries(queries_csv)
    if not queries:
        logger.error("No queries to process. Exiting the process.")
        return

    # List to store all results
    all_results = []

    # Process each query
    for query_id, query_text in queries:
        logger.info(f"Processing Query ID {query_id}: {query_text}")
        top_docs = retrieve_documents(query_text, inverted_index, idf, total_documents)
        for doc, score in top_docs:
            all_results.append((query_id, doc))
        logger.info(f"Completed processing Query ID {query_id}.")

    # Save all results to the output CSV file
    save_results_to_csv(all_results, output_csv)
    logger.info("Document retrieval process completed successfully.")

if __name__ == '__main__':
    main()
