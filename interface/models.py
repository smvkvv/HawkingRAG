import yaml
from pgvector.sqlalchemy import Vector
from sqlalchemy import Column, Date, ForeignKey, Integer, String, Text

from interface.database import Base


# Load configuration to access vector dimension
config_path = "interface/config.yml"
with open(config_path, "r") as file:
    config = yaml.safe_load(file)


class ParagraphDataset(Base):
    __tablename__ = "paragraph_dataset"

    id = Column(Integer, primary_key=True, index=True)
    dt = Column(Date, nullable=False)
    chapter = Column(String, nullable=False)
    paragraph_text = Column(Text, nullable=False)


class DataChunks(Base):
    __tablename__ = "data_chunks"

    id = Column(Integer, primary_key=True, index=True)
    parent_id = Column(Integer, ForeignKey("paragraph_dataset.id"), nullable=False)
    dt = Column(Date, nullable=False)
    chapter = Column(String, nullable=False)
    chunk_text = Column(Text, nullable=False)
    vector = Column(Vector(config["embedding_model"]["dimension"]))

    def __init__(self, parent_id, dt, chapter, chunk_text, vector):
        self.parent_id = parent_id
        self.dt = dt
        self.chapter = chapter
        self.chunk_text = chunk_text
        self.vector = vector
