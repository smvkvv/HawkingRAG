import os
import json
import string
import logging
import secrets
from dotenv import load_dotenv
from datetime import datetime
from pydoc import locate
from typing import Dict, List, Union
from gigachat import GigaChat
from gigachat.models import Chat, Messages, MessagesRole
from pathlib import Path

from opensearchpy import OpenSearch
from tqdm import tqdm

from interface.chunker import AbstractBaseChunker
from interface.database import SessionLocal
from interface.elastic import create_index, update_search
from interface.embedder import Embedder
from interface.models import DataChunks, ParagraphDataset
from interface.schemas import EmbedderSettings, Context


logger = logging.getLogger(__name__)


def initialize_llm_client(config: Dict) -> GigaChat:
    """
    Initializes and returns the LLM client using provided configuration.
    """
    try:
        load_dotenv('.env')
        llm_client = GigaChat(credentials=os.environ.get("LLM_API_KEY"), verify_ssl_certs=False)
        logger.info("LLM client initialized successfully.")
        return llm_client
    except Exception as e:
        logger.error(f"Failed to initialize LLM client: {e}")
        raise

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


def load_data(embedder: Embedder, os_client: OpenSearch, config: Dict) -> None:
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
            load_and_process_text_documents(db, embedder, os_client, config)

        logger.info("Data loading process completed successfully.")
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        db.rollback()
        raise
    finally:
        db.close()


def load_and_process_text_documents(db, embedder: Embedder, os_client: OpenSearch, config: Dict) -> None:
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
        create_index(index_name="chunks", os_client=os_client)
        chapter: str = "nan"

        for paragraph in tqdm(paragraphs):
            if paragraph.strip() in chapter_names:
                chapter: str = paragraph
            else:
                doc = ParagraphDataset(dt=datetime.now(), chapter=chapter, paragraph_text=paragraph)
                db.add(doc)
                db.commit()

                chunks = chunker.chunk_text(paragraph)
                store_chunks(db, os_client, doc, chunks, embedder)
        logger.info(f"Processed and stored chunks from {file_path}")
    except Exception as e:
        logger.error(f"Error processing text documents: {e}")
        raise


def store_chunks(db, os_client: OpenSearch, doc: ParagraphDataset, chunks: List[str], embedder: Embedder) -> None:
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
        update_search(doc=doc, chunks=chunks, os_client=os_client)
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


def retrieve_fulltext_search(os_client: OpenSearch, config: Dict, question: str) -> List[Context]:
    top_k = config["retrieval"]["top_k_fulltext"]
    query: Dict = {"query": {"match": {"text": {"query": question}}}, "size": top_k}
    response: Dict = os_client.search(index=config["elastic_params"]["index_name"], body=query)

    return [Context(text=hit["_source"]['text'], chapter=hit["_source"]['chapter'], type='fulltext') for hit in response['hits']['hits']]


def retrieve_contexts(query: str, embedder: Embedder, config: Dict, os_client: OpenSearch) -> List[Context]:
    """
    Retrieves the most relevant contexts from DataChunks for a given query using vector search.
    """
    db = SessionLocal()
    top_chunks = []
    try:
        if config["retrieval"]["vector_search_enabled"]:
            top_chunks.extend(retrieve_semantic_search(db, query, embedder, config))
        if config["retrieval"]["fulltext_search_enabled"]:
            top_chunks.extend(retrieve_fulltext_search(os_client, config, query))

        # Add deduplication

        result = top_chunks[:config["retrieval"]["top_k"]]
        logger.info(f"Retrieved top {len(result)} contexts for the query")
        return result
    except Exception as e:
        logger.error(f"Error retrieving contexts: {e}")
        raise
    finally:
        db.close()

def generate_response(llm_client, contexts, query, config):
    """
    Generates a response based on retrieved contexts and the input query.
    """
    try:
        prompt = build_prompt(contexts, query)
        response = llm_client.stream(
            Chat(
                messages=[
                    Messages(role=MessagesRole.SYSTEM, content=config["llm"]["system_prompt"]),
                    Messages(role=MessagesRole.USER, content=prompt),
                ],
                temperature=config["llm"]["temperature"],
                top_p=config["llm"]["top_p"],
                max_tokens=config["llm"]["max_tokens"],
            )
        )

        generated_response = ""
        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                generated_response += chunk.choices[0].delta.content

        logger.info(f"Generated response: {generated_response[:30]}...")
        return generated_response
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        raise

def build_prompt(contexts: List[str], query: str) -> str:
    """
    Constructs the prompt for the LLM based on the given contexts and query.
    """

    prompt = "Отвечай используя контекст:\n"
    for i, context in enumerate(contexts):
        prompt += f"Контекст {i + 1}: {context}\n"
    prompt += f"Вопрос: {query}\nНе упоминай, что ты пользуешься контекстом\nПодробный Ответ: "
    return prompt


def rewrite_query(llm_client, query, config):
    """
    Answers user's query using LLM_rewriter
    """
    try:
        response = llm_client.stream(
            Chat(
                messages=[
                    Messages(role=MessagesRole.SYSTEM, content=config["llm_rewriter"]["system_prompt"]),
                    Messages(role=MessagesRole.USER, content=f"Вопрос: {query}"),
                ],
                temperature=config["llm_rewriter"]["temperature"],
                top_p=config["llm_rewriter"]["top_p"],
                max_tokens=config["llm_rewriter"]["max_tokens"],
            )
        )

        rewrited_query = ""
        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                rewrited_query += chunk.choices[0].delta.content

        rewrited_query += f'\n------------------\n{query}'
        logger.info(f"Rewrited query: {rewrited_query}")
        return rewrited_query
    except Exception as e:
        logger.error(f"Error rewriting query: {e}")
        raise

def is_prompt_injection(message: str, llm_client: GigaChat, config) -> bool:
    security_key = ''.join(secrets.choice(string.ascii_letters + string.digits) for _ in range(16))
    canary_prompt = config["security"]["canary_prompt"].format(
        message=message.replace('"', r'\"'),
        security_key=security_key,
    )
    try:
        response = llm_client.chat(
            Chat(
                messages=[
                    Messages(role=MessagesRole.USER, content=canary_prompt),
                ],
                temperature=0.0,  # Полная детерминированность
                max_tokens=200
            )
        )
        answer = response.choices[0].message.content.strip()
        try:
            returned_json = json.loads(answer)
            # Проверяем точное соответствие
            expected = {
                "message": message,
                "key": security_key
            }
            return returned_json != expected
        except json.JSONDecodeError:
            return True  # Не смогли распарсить - инъекция
    except Exception as e:
        print(f"Prompt injection check failed: {e}")
        return True  # Fail secure

def process_request(config: dict, embedder: Embedder, llm_client: GigaChat, query: str, os_client: OpenSearch) -> Union[
    dict, str]:
    """
    Processes the incoming query by retrieving relevant contexts and generating a response.
    """
    try:
        canary_check = is_prompt_injection(query, llm_client, config)
        if (canary_check == False):
            return {"response": config["security"]["canary_check_error_message"], "context": list()}
        rewrited_query = rewrite_query(llm_client, query, config)
        contexts: List[Context] = retrieve_contexts(
            rewrited_query, embedder, config, os_client
        ) # В ретривер отправляется переписанный запрос
        
        # Generate the response
        llm_response = generate_response(llm_client, contexts, query, config)

        # Return both the response and the contexts used
        return {"response": llm_response, "context": contexts}

    except Exception as e:
        logger.error(f"Failed to process request: {e}")
        return (
            "An error occurred while processing your request. Please try again later."
        )
