import uuid
from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Index, Integer, JSON, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR
from sqlalchemy.orm import relationship

try:
    from pgvector.sqlalchemy import Vector
except ImportError:
    Vector = None

from app.database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(120), unique=True, nullable=True)
    full_name = Column(String(100), nullable=True)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    conversations = relationship(
        "Conversation",
        back_populates="user",
        cascade="all, delete-orphan",
        order_by="desc(Conversation.updated_at)",
    )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "username": self.username,
            "email": self.email,
            "full_name": self.full_name,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()), index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    title = Column(String(255), nullable=False, default="New Chat")
    is_archived = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
        index=True,
    )

    user = relationship("User", back_populates="conversations")
    messages = relationship(
        "ChatMessage",
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="ChatMessage.created_at",
    )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "title": self.title,
            "is_archived": self.is_archived,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "message_count": len(self.messages) if self.messages is not None else 0,
        }


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()), index=True)
    conversation_id = Column(
        String(36),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=True)
    role = Column(String(20), nullable=False)  # 'user', 'assistant', 'system'
    content = Column(Text, nullable=False)
    sources = Column(JSON, nullable=True)  # List of source objects/citations
    token_count = Column(Integer, nullable=True)
    is_error = Column(Boolean, default=False, nullable=False)
    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        index=True,
    )

    conversation = relationship("Conversation", back_populates="messages")

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "conversation_id": self.conversation_id,
            "user_id": self.user_id,
            "role": self.role,
            "content": self.content,
            "sources": self.sources or [],
            "token_count": self.token_count,
            "is_error": self.is_error,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class Document(Base):
    __tablename__ = "documents"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()), index=True)
    ticker = Column(String(10), nullable=False, index=True)  # e.g. AAPL, TSLA
    company_name = Column(String(255), nullable=False)  # e.g. Apple Inc., Tesla, Inc.
    filing_type = Column(String(20), nullable=False, default="10-K")  # 10-K, 10-Q
    fiscal_year = Column(Integer, nullable=False, index=True)  # e.g. 2025
    period_end_date = Column(String(20), nullable=True)  # e.g. 2025-09-27
    source_filename = Column(String(255), nullable=False)  # e.g. aapl-20250927.htm
    total_chunks = Column(Integer, default=0, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "ticker": self.ticker,
            "company_name": self.company_name,
            "filing_type": self.filing_type,
            "fiscal_year": self.fiscal_year,
            "period_end_date": self.period_end_date,
            "source_filename": self.source_filename,
            "total_chunks": self.total_chunks,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()), index=True)
    document_id = Column(String(36), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True)
    part = Column(String(20), nullable=True, index=True)  # 'PART I', 'PART II'
    item = Column(String(20), nullable=True, index=True)  # 'Item 1A', 'Item 7', 'Item 8'
    sub_section = Column(Text, nullable=True)
    breadcrumb = Column(Text, nullable=False)
    chunk_type = Column(String(20), nullable=False, index=True)  # 'narrative', 'table'
    chunk_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)  # Clean markdown or text for LLM
    metadata_ = Column("metadata", JSONB, nullable=True)  # Structural metadata, dimensions, units
    content_tsv = Column(TSVECTOR, nullable=True)  # For full-text search
    embedding = Column(Vector(1536) if Vector is not None else Text, nullable=True)  # Vector embedding
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    document = relationship("Document", back_populates="chunks")

    __table_args__ = (
        Index("idx_chunks_doc_item", "document_id", "item"),
        Index("idx_chunks_tsv", "content_tsv", postgresql_using="gin"),
        Index("idx_chunks_embedding_hnsw", "embedding", postgresql_using="hnsw", postgresql_with={"m": 16, "ef_construction": 64}, postgresql_ops={"embedding": "vector_cosine_ops"}) if Vector is not None else Index("idx_chunks_doc_idx", "document_id", "chunk_index"),
    )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "document_id": self.document_id,
            "part": self.part,
            "item": self.item,
            "sub_section": self.sub_section,
            "breadcrumb": self.breadcrumb,
            "chunk_type": self.chunk_type,
            "chunk_index": self.chunk_index,
            "content": self.content,
            "metadata": self.metadata_ or {},
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }

