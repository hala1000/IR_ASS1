import os
import re
import csv
import logging
import pandas as pd
import nltk
import shutil
from collections import defaultdict
from nltk.corpus import stopwords, wordnet, words as nltk_words
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from multiprocessing import Pool, cpu_count
from typing import Optional, Tuple, List, Dict
from pathlib import Path


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("text_processing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize global variables
stop_words = set(stopwords.words('english'))  # Load English stop words
lemmatizer = WordNetLemmatizer()  # Initialize a lemmatizer for word normalization
english_words = set(nltk_words.words())  # Create a set of English words for quick lookup

def initialize_nltk_resources():
    """
    Downloads necessary NLTK resources if they are not already present.
    """
    resources = [
        ('tokenizers/punkt', 'punkt'),
        ('corpora/stopwords', 'stopwords'),
        ('taggers/averaged_perceptron_tagger', 'averaged_perceptron_tagger'),
        ('corpora/wordnet', 'wordnet'),
        ('corpora/words', 'words')
    ]
    for path, name in resources:
        try:
            nltk.data.find(path)
            logger.info(f"NLTK resource '{name}' already exists.")
        except LookupError:
            nltk.download(name)
            logger.info(f"Downloaded NLTK resource '{name}'.")

def clean_text(text: str) -> str:
    """
    Cleans the input text by removing numbers, punctuation, single-character words, and extra spaces.

    Args:
        text (str): The input text to clean.

    Returns:
        str: Cleaned text string.
    """
    logger.debug("Cleaning text.")
    text = re.sub(r'\d+', ' ', text)  # Remove numbers
    text = re.sub(r'[^\w\s]', ' ', text)  # Remove punctuation
    text = re.sub(r'\b\w\b', ' ', text)  # Remove single-character words
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

def tokenize_text(text: str) -> List[str]:
    """
    Tokenizes the input text into individual words.

    Args:
        text (str): The text to tokenize.

    Returns:
        list: List of word tokens.
    """
    logger.debug("Tokenizing text.")
    return word_tokenize(text)

def normalize_tokens(tokens: List[str]) -> List[str]:
    """
    Converts all tokens to lowercase.

    Args:
        tokens (list): List of tokens to normalize.

    Returns:
        list: List of lowercase tokens.
    """
    logger.debug("Normalizing tokens to lowercase.")
    return [token.lower() for token in tokens]

def remove_stopwords_from_tokens(tokens: List[str]) -> List[str]:
    """
    Removes English stopwords from the list of tokens.

    Args:
        tokens (list): List of tokens to filter.

    Returns:
        list: List of tokens without stopwords.
    """
    logger.debug("Removing stopwords.")
    return [word for word in tokens if word not in stop_words]

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
        tokens (list): List of tokens to lemmatize.

    Returns:
        list: List of lemmatized words.
    """
    logger.debug("Lemmatizing tokens.")
    pos_tags = pos_tag(tokens)
    return [
        lemmatizer.lemmatize(word, get_wordnet_pos(tag))
        for word, tag in pos_tags
        if word.isalpha() and len(word) > 1
    ]

def remove_non_english_words_from_tokens(tokens: List[str]) -> List[str]:
    """
    Removes any words that are not in the English dictionary.

    Args:
        tokens (list): List of word tokens.

    Returns:
        list: List of English words.
    """
    logger.debug("Removing non-English words.")
    return [word for word in tokens if word in english_words]

def process_text(text: str) -> List[str]:
    """
    Combines all text preprocessing steps: cleaning, tokenizing, normalizing, removing stopwords, lemmatizing,
    and removing non-English words.

    Args:
        text (str): The raw text to process.

    Returns:
        list: List of processed tokens.
    """
    logger.info("Processing text.")
    text = clean_text(text)
    tokens = tokenize_text(text)
    tokens = normalize_tokens(tokens)
    tokens = remove_stopwords_from_tokens(tokens)
    tokens = lemmatize_tokens(tokens)
    #tokens = remove_non_english_words_from_tokens(tokens)
    return tokens

def process_file(file_path: str) -> Tuple[List[str], Dict[str, int], int]:
    """
    Reads a file and applies the full text processing pipeline.

    Args:
        file_path (str): Path to the file to process.

    Returns:
        tuple: (processed_words, word_freq, total_words)
    """
    logger.info(f"Processing file: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {e}")
        return [], {}, 0

    # Process the content using the process_text function
    processed_words = process_text(content)

    # Count word frequencies in the current document
    word_freq = defaultdict(int)
    for word in processed_words:
        word_freq[word] += 1

    total_words = len(processed_words)

    return processed_words, word_freq, total_words

def save_inverted_index_to_csv(inverted_index: Dict[str, Dict[str, float]], csv_file_path: str):
    """
    Saves the inverted index dictionary to a CSV file.

    Args:
        inverted_index (dict): Dictionary containing words as keys and dict of documents and tf as values.
        csv_file_path (str): Path to save the CSV file.
    """
    logger.info(f"Saving inverted index to CSV file: {csv_file_path}")
    try:
        with open(csv_file_path, 'w', newline='', encoding='utf-8') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['Word', 'Documents'])
            for word, doc_tfs in sorted(inverted_index.items()):
                doc_tfs_str = ', '.join([f'{doc}:{tf:.6f}' for doc, tf in doc_tfs.items()])
                writer.writerow([word, doc_tfs_str])
        logger.info(f"Inverted index successfully saved to {csv_file_path}")
    except Exception as e:
        logger.error(f"Error saving inverted index to {csv_file_path}: {e}")

def create_inverted_index(
    current_directory: str,
    directory_path: str,
    processed_directory: str,
    inverted_directory: str,
    chunk_size: int = 2000
):
    """
    Processes files in batches, creating an inverted index where each word maps to the files containing it.
    Saves each batch as a CSV file for memory management.

    Args:
        current_directory (str): Path to the current working directory.
        directory_path (str): Path to the directory with text files.
        processed_directory (str): Path for saving processed files.
        inverted_directory (str): Path for saving inverted index CSV files.
        chunk_size (int): Number of files to process per batch.
    """
    logger.info("Creating inverted index.")
    file_count = 0  # Counter for processed files
    chunk_index = 0  # Index for naming chunk files

    # Validate directories
    if not os.path.isdir(current_directory):
        logger.error(f"The current directory '{current_directory}' does not exist.")
        return

    if not os.path.isdir(directory_path):
        logger.error(f"The directory path '{directory_path}' does not exist.")
        return

    # Delete and recreate the processed_directory
    if os.path.exists(processed_directory):
        shutil.rmtree(processed_directory)
        logger.info(f"Deleted existing processed directory at '{processed_directory}'.")
    os.makedirs(processed_directory)
    logger.info(f"Created processed directory at '{processed_directory}'.")

    # Delete and recreate the inverted_directory
    if os.path.exists(inverted_directory):
        shutil.rmtree(inverted_directory)
        logger.info(f"Deleted existing inverted index directory at '{inverted_directory}'.")
    os.makedirs(inverted_directory)
    logger.info(f"Created inverted index directory at '{inverted_directory}'.")

    # Gather all text files
    files = [
        os.path.join(directory_path, filename)
        for filename in os.listdir(directory_path)
        if filename.endswith(".txt")
    ]

    logger.info(f"Found {len(files)} text files to process.")

    for i in range(0, len(files), chunk_size):
        chunk = files[i:i + chunk_size]
        logger.info(f"Processing chunk {chunk_index + 1} with {len(chunk)} files.")

        # Initialize inverted index for the current chunk
        inverted_index = defaultdict(lambda: defaultdict(float))

        # Process files in parallel using multiprocessing
        with Pool(cpu_count()) as pool:
            results = pool.map(process_file, chunk)

        for file_idx, result in enumerate(results):
            processed_words, word_freq, total_words = result
            if total_words == 0:
                logger.warning(f"No words processed for file: {chunk[file_idx]}")
                continue

            filename = os.path.basename(chunk[file_idx])
            file_count += 1

            # Calculate tf for each word in the current document
            for word, count in word_freq.items():
                tf = count / total_words
                inverted_index[word][filename] = tf

            # Save processed words to the processed_directory
            processed_file_path = os.path.join(processed_directory, filename)
            try:
                with open(processed_file_path, 'w', encoding='utf-8') as processed_file:
                    processed_file.write(' '.join(processed_words))
            except Exception as e:
                logger.error(f"Error writing processed file {processed_file_path}: {e}")

        logger.info(f"Processed {file_count} file(s) so far.")

        # Save the current chunk's inverted index to a CSV file
        csv_filename = f'inverted_index_chunk_{chunk_index}.csv'
        csv_file_path = os.path.join(inverted_directory, csv_filename)
        save_inverted_index_to_csv(inverted_index, csv_file_path)

        chunk_index += 1

    logger.info("All files have been processed successfully.")

def merge_inverted_indexes(
    inverted_index_directory: str,
    output_csv_filename: str,
    top_n: int = 10
):
    """
    Merges all chunked inverted index CSV files into a single CSV file containing the complete inverted index.
    Retains only the top N documents with the highest tf values for each word.

    Args:
        inverted_index_directory (str): Path to the directory containing chunked inverted index CSV files.
        output_csv_filename (str): Path to save the merged inverted index CSV file.
        top_n (int): Number of top documents to retain for each word based on tf values.
    """
    logger.info("Merging all inverted indexes into a single CSV file.")
    chunks_list = []

    # Iterate over all CSV files in the inverted_index_directory
    for filename in os.listdir(inverted_index_directory):
        if filename.endswith('.csv'):
            csv_file_path = os.path.join(inverted_index_directory, filename)
            logger.debug(f"Processing chunk file: {csv_file_path}")
            try:
                for chunk in pd.read_csv(csv_file_path, chunksize=100000):
                    # Split 'Documents' into individual document-tf pairs
                    chunk_expanded = chunk.set_index('Word')['Documents'].str.split(', ').explode().reset_index()
                    chunk_expanded[['Document', 'tf']] = chunk_expanded['Documents'].str.split(':', expand=True)
                    chunk_expanded['tf'] = chunk_expanded['tf'].astype(float)
                    chunk_expanded = chunk_expanded.drop(columns=['Documents'])
                    chunks_list.append(chunk_expanded)
                    logger.debug(f"Processed a chunk from {filename}.")
            except Exception as e:
                logger.error(f"Error processing file {csv_file_path}: {e}")

    if not chunks_list:
        logger.error("No data found to merge. Exiting merge process.")
        return

    # Concatenate all chunks into a single DataFrame
    logger.info("Concatenating all chunks into a single DataFrame.")
    aggregated_df = pd.concat(chunks_list, ignore_index=True)

    # Group by 'Word' and 'Document' to sum tf values
    logger.info("Aggregating tf values for each Word-Document pair.")
    aggregated_df = aggregated_df.groupby(['Word', 'Document'], as_index=False)['tf'].sum()

    # Sort by 'Word' ascending and 'tf' descending
    logger.info("Sorting the aggregated DataFrame.")
    aggregated_df = aggregated_df.sort_values(['Word', 'tf'], ascending=[True, False])

    # Retain only the top N documents per word
    logger.info(f"Retaining only the top {top_n} documents per word based on tf values.")
    aggregated_df = aggregated_df.groupby('Word').head(top_n)

    # Combine 'Document' and 'tf' into 'Doc_tf' column
    logger.info("Combining Document and tf into a single column.")
    aggregated_df['Doc_tf'] = aggregated_df['Document'] + ':' + aggregated_df['tf'].astype(str)

    # Group by 'Word' and aggregate 'Doc_tf' as comma-separated strings
    logger.info("Grouping Doc_tf values for each Word.")
    final_df = aggregated_df.groupby('Word')['Doc_tf'].apply(', '.join).reset_index()
    final_df.rename(columns={'Doc_tf': 'Documents'}, inplace=True)

    # Save the final merged inverted index to CSV
    try:
        final_df.to_csv(output_csv_filename, index=False)
        logger.info(f"Successfully saved merged inverted index to '{output_csv_filename}'.")
    except Exception as e:
        logger.error(f"Error saving merged inverted index to '{output_csv_filename}': {e}")

def main():
    """
    Main function to execute the script.
    """
    logger.info("Starting text processing and inverted index creation.")

    # Initialize NLTK resources
    initialize_nltk_resources()

    # Define directory paths
    current_directory = Path("../")
    directory_path = Path("../data/full_docs")  # Directory containing text files
    processed_directory = Path("../processed_docs_big_ranked_non_en")  # Directory to save processed files
    inverted_index_directory = Path("../inverted_index_chunks_big_ranked_non_en")  # Directory to save inverted index chunks

    # Create inverted index from text files
    create_inverted_index(
        current_directory=current_directory,
        directory_path=directory_path,
        processed_directory=processed_directory,
        inverted_directory=inverted_index_directory,
        chunk_size=2000  # Adjust chunk size as needed
    )

    # Merge all inverted index chunks into a single CSV file
    output_csv_filename = '../merged_inverted_index_pandas_big_ranked_non_en.csv'
    merge_inverted_indexes(
        inverted_index_directory=inverted_index_directory,
        output_csv_filename=output_csv_filename,
        top_n=10  # Retain top 10 documents per word
    )

    logger.info("Text processing and inverted index creation completed successfully.")

if __name__ == '__main__':
    main()
