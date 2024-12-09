import logging
from typing import Any, Dict, Iterator, List

import more_itertools
from elasticsearch import Elasticsearch, ElasticsearchException
from elasticsearch.helpers import bulk

from interface.models import ParagraphDataset

logger = logging.getLogger(__name__)


def create_index(index_name: str, es_client: Elasticsearch) -> None:
    mapping: Dict = {
        "mappings": {
            "properties": {
                "text": {"type": "text"},
                "chapter": {"type": "keyword"}
            }
        }
    }

    if not es_client.indices.exists(index=index_name):
        es_client.indices.create(index=index_name, body=mapping)
        logger.info(f"Successfully created index {index_name}")


def load(chunks: List[str]) -> Iterator[Any]:
    for chunk in chunks:
        try:
            yield generate_document_source(chunk)
        except Exception:
            raise


def generate_document_source(chunk: str) -> Dict[str, str]:
    return {"text": chunk}


def update_search(doc: ParagraphDataset, chunks: List[str], es_client: Elasticsearch, batch_size: int = 500) -> None:
    total_inserted_docs: int = 0
    total_errors: int = 0

    for chunk in more_itertools.ichunked(load(chunks), batch_size):
        bucket_data = []
        for document in chunk:
            cur = {
                "_index": "chunks",
                "_source": document,
            }
            cur['_source']['chapter'] = doc.chapter
            bucket_data.append(cur)
        try:
            inserted, errors = bulk(es_client, bucket_data, max_retries=4, raise_on_error=False)
            errors_num = len(errors) if isinstance(errors, list) else errors  # type: ignore
            logger.debug(f"{inserted} docs successfully inserted by bulk with {errors_num} errors")
            total_inserted_docs += inserted
            total_errors += errors_num
            if isinstance(errors, list):  # type: ignore
                for error in errors:  # type: ignore
                    logger.error(f"Doc was not inserted with error: {error}")
        except ElasticsearchException as e:
            logger.exception(f"Error while pushing data to elasticsearch: {e}")
            raise
