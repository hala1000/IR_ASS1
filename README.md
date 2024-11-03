# Document Retrieval System

## Overview
The Document Retrieval System is designed to process a large collection of text documents, create an inverted index for efficient searching, and retrieve relevant documents based on user queries. The system employs natural language processing techniques and the Vector Space Model to enhance the accuracy and relevance of search results.

## Key Components
- **Text Processing:** Cleans, tokenizes, normalizes, and lemmatizes text data to prepare it for indexing and retrieval.
- **Inverted Index Creation:** Constructs an inverted index that maps terms to their occurrences in the document collection, along with term frequency (TF) information.
- **Document Retrieval:** Implements the Vector Space Model to rank documents based on their relevance to user queries using TF-IDF scoring and cosine similarity.

## Usage
1.	Prepare Your Data:
Place your text documents in the data/full_docs directory.
2.	Run the Text Processing and Index Creation:
Execute the main script to process the documents and create the inverted index:

python 1_text_processing_and _index_creation.py

3.	Query the System:
To retrieve documents based on queries, modify the query file path in the code and run the retrieval script:

python 2_vsm_retrieval.py

4.	Evaluate the Results:
Use the evaluation script to calculate metrics such as Mean Average Precision (MAP) and Mean Average Recall (MAR):

python 3_vsm_evaluation.py

Metrics

•	Mean Average Precision (MAP): Measures the accuracy of the retrieval system.
•	Mean Average Recall (MAR): Evaluates the system’s ability to retrieve all relevant documents.
