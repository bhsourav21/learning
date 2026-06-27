# RAG Pipeline Notes

## Overview
RAG (Retrieval-Augmented Generation) combines a retriever
with a generative model to answer questions grounded in docs.

## Key Components
- **Document Loader** -- ingests raw sources
- **Text Splitter** -- breaks docs into chunks
- **Embeddings** -- converts text to vectors
- **Vector Store** -- stores and retrieves vectors

## Useful Links
See the [LangChain docs](https://python.langchain.com).