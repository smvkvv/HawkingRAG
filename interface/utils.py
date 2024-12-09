import logging
from datetime import datetime
from pydoc import locate
from typing import Dict, List, Union
from pathlib import Path

from elasticsearch import Elasticsearch
from tqdm import tqdm

from interface.chunker import AbstractBaseChunker
from interface.database import SessionLocal
from interface.elastic import create_index, update_search
from interface.embedder import Embedder
from interface.models import DataChunks, ParagraphDataset
from interface.schemas import EmbedderSettings, Context


logger = logging.getLogger(__name__)



def initialize_embedding_model(config: Dict) -> Embedder:
    """
    Initializes and returns the embedding model using provided configuration.
    """
    try:
        settings = EmbedderSettings(**config["embedding_model"])
        embedder = Embedder(settings)
        logger.info(f"Initialized embedding model {settings.model_name}")
        return embedder
    except Exception as e:
        logger.error(f"Failed to initialize embedding model: {e}")
        raise


def load_data(embedder: Embedder, es_client: Elasticsearch, config: Dict) -> None:
    """
    Loads and processes data into the database by chunking text and vectorizing it.
    Only loads data into a table if the table is empty.
    """
    db = SessionLocal()
    try:
        # Check if there is any data in the ParagraphDataset table
        if db.query(ParagraphDataset).first() is not None:
            logger.info(
                "Data already exists in the ParagraphDataset table. Skipping data loading for this table."
            )
        else:
            logger.info(
                "No existing data found in ParagraphNpaDataset. Proceeding with data loading and processing for this table."
            )
            load_and_process_text_documents(db, embedder, es_client, config)

        logger.info("Data loading process completed successfully.")
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        db.rollback()
        raise
    finally:
        db.close()


def load_and_process_text_documents(db, embedder: Embedder, es: Elasticsearch, config: Dict) -> None:
    """
    Loads and processes text documents from a file, chunking and vectorizing the content.
    """

    chunker_cls = locate(config["data_processing"]["chunker"]["py_class"])
    chunker: AbstractBaseChunker = chunker_cls(**config["data_processing"]["chunker"]["kwargs"])

    try:
        file_path: Path = Path(config["data_sources"]["text_file"])
        chapter_names: list[str] = config["brief_history"]["chapter_names"]
        first_paragraph_num: int = config["brief_history"]["first_paragraph_num"]

        with file_path.open() as file:
            paragraphs: list[str] = [line for line in file if line.strip()][first_paragraph_num:]
        create_index(index_name="chunks", es_client=es)
        chapter: str = "nan"

        for paragraph in tqdm(paragraphs):
            if paragraph.strip() in chapter_names:
                chapter: str = paragraph
            else:
                doc = ParagraphDataset(dt=datetime.now(), chapter=chapter, paragraph_text=paragraph)
                db.add(doc)
                db.commit()

                chunks = chunker.chunk_text(paragraph)
                store_chunks(db, es, doc, chunks, embedder)
        logger.info(f"Processed and stored chunks from {file_path}")
    except Exception as e:
        logger.error(f"Error processing text documents: {e}")
        raise


def store_chunks(db, es: Elasticsearch, doc: ParagraphDataset, chunks: List[str], embedder: Embedder) -> None:
    """
    Vectorizes and stores text chunks in the database.
    """
    try:
        embeddings = embedder.encode(chunks, doc_type="document")

        for passage, embedding in zip(chunks, embeddings, strict=True):
            db_chunk = DataChunks(parent_id=doc.id, dt=doc.dt,
                                  chapter=doc.chapter,
                                  chunk_text=passage,
                                  vector=embedding)
            db.add(db_chunk)
        db.commit()
        update_search(doc=doc, chunks=chunks, es_client=es)
    except Exception as e:
        logger.error(f"Error storing chunks in the database: {e}")
        raise


def retrieve_semantic_search(db, query: str, embedder: Embedder, config: Dict) -> List[Context]:
    query_vector = embedder.encode([query], doc_type="query")[0].tolist()

    # Define parameters
    k = config["retrieval"]["top_k_vector"]
    similarity_threshold = config["retrieval"]["similarity_threshold"]

    # Query the database for the most similar contexts based on cosine similarity
    results = (
        db.query(
            DataChunks,
            DataChunks.vector.cosine_distance(query_vector).label("distance"),
        )
        .filter(
            DataChunks.vector.cosine_distance(query_vector) < similarity_threshold
        )
        .order_by("distance")
        .limit(k)
        .all()
    )
    return [Context(text=result.DataChunks.chunk_text, chapter=result.DataChunks.chapter, type='semantic') for result in results]


def retrieve_fulltext_search(es_client: Elasticsearch, config: Dict, question: str) -> List[Context]:
    top_k = config["retrieval"]["top_k_fulltext"]
    query: Dict = {"query": {"match": {"text": {"query": question}}}, "size": top_k}
    response: Dict = es_client.search(index=config["elastic_params"]["index_name"], body=query)

    return [Context(text=hit["_source"]['text'], chapter=hit["_source"]['chapter'], type='fulltext') for hit in response['hits']['hits']]


def retrieve_contexts(query: str, embedder: Embedder, config: Dict, es: Elasticsearch) -> List[Context]:
    """
    Retrieves the most relevant contexts from DataChunks for a given query using vector search.
    """
    db = SessionLocal()
    top_chunks = []
    try:
        if config["retrieval"]["vector_search_enabled"]:
            top_chunks.extend(retrieve_semantic_search(db, query, embedder, config))
        if config["retrieval"]["fulltext_search_enabled"]:
            top_chunks.extend(retrieve_fulltext_search(es, config, query))

        # Add deduplication

        result = top_chunks[:config["retrieval"]["top_k"]]
        logger.info(f"Retrieved top {len(result)} contexts for the query")
        return result
    except Exception as e:
        logger.error(f"Error retrieving contexts: {e}")
        raise
    finally:
        db.close()


def process_request(config: dict, embedder: Embedder, query: str, es: Elasticsearch) -> Union[
    dict, str]:
    """
    Processes the incoming query by retrieving relevant contexts and generating a response.
    """
    try:
        contexts: List[Context] = retrieve_contexts(query, embedder, config, es)

        return {"context": contexts}

    except Exception as e:
        logger.error(f"Failed to process request: {e}")
        return (
            "An error occurred while processing your request. Please try again later."
        )