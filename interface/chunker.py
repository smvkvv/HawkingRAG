from abc import ABC, abstractmethod

from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


class AbstractBaseChunker(ABC):
    @abstractmethod
    def chunk(self, documents: list[str]) -> list[str]:
        pass

    @abstractmethod
    def chunk_text(self, text: str) -> list[str]:
        pass


class RecursiveCharacterTextSplitterChunker(AbstractBaseChunker):
    def __init__(self, chunk_size, chunk_overlap, separators=None):
        super().__init__()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            add_start_index=True,
            is_separator_regex=False,
            separators=separators,
        )

    def chunk(self, documents: list[Document]) -> list[Document]:
        return self.text_splitter.split_documents(documents)

    def chunk_text(self, text: str) -> list[str]:
        return self.text_splitter.split_text(text)


class SemanticTextChunker(AbstractBaseChunker):
    def __init__(
        self,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=None,
        number_of_chunks=None,
    ):
        self.splitter = SemanticChunker(
            OpenAIEmbeddings(),
            breakpoint_threshold_type=breakpoint_threshold_type,
            breakpoint_threshold_amount=breakpoint_threshold_amount,
            number_of_chunks=number_of_chunks,
        )

    def chunk(self, documents: list[Document]) -> list[Document]:
        return self.splitter.split_documents(documents)

    def chunk_text(self, text: str) -> list[str]:
        return self.splitter.split_text(text)


class AbstractTransformChunk(ABC):
    @abstractmethod
    def transform(self, chunks: list[str], text: str) -> list[str]:
        pass
