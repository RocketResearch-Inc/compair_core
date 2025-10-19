from __future__ import annotations

import binascii
import hashlib
import os
import secrets
from datetime import datetime, timezone
from math import sqrt
from typing import Sequence
from uuid import uuid4

try:  # Optional: only required when using pgvector backend
    from pgvector.sqlalchemy import Vector
except ImportError:  # pragma: no cover - optional dependency in core
    Vector = None  # type: ignore[assignment]

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Identity,
    Integer,
    JSON,
    String,
    Table,
    Text,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    MappedAsDataclass,
    mapped_column,
    relationship,
)

_EDITION = os.getenv("COMPAIR_EDITION", "core").lower()
_DEFAULT_DIM = 1536 if _EDITION == "cloud" else 384
_DIM_ENV = (
    os.getenv("COMPAIR_EMBEDDING_DIM")
    or os.getenv("COMPAIR_EMBEDDING_DIMENSION")
    or os.getenv("COMPAIR_LOCAL_EMBED_DIM")
    or str(_DEFAULT_DIM)
)

try:
    EMBEDDING_DIMENSION = int(_DIM_ENV)
except ValueError:  # pragma: no cover - invalid configuration
    EMBEDDING_DIMENSION = _DEFAULT_DIM


def _detect_vector_backend() -> str:
    explicit = os.getenv("COMPAIR_VECTOR_BACKEND")
    if explicit:
        return explicit.lower()

    db = os.getenv("DB")
    db_user = os.getenv("DB_USER")
    db_passw = os.getenv("DB_PASSW")
    db_url = os.getenv("DB_URL")
    database_url = os.getenv("DATABASE_URL", "")

    if all([db, db_user, db_passw, db_url]):
        return "pgvector"
    if database_url.lower().startswith(("postgres://", "postgresql://")):
        return "pgvector"
    return "json"


VECTOR_BACKEND = _detect_vector_backend()


def _embedding_column():
    if VECTOR_BACKEND == "pgvector":
        if Vector is None:
            raise RuntimeError(
                "pgvector is required when COMPAIR_VECTOR_BACKEND is set to 'pgvector'."
            )
        return mapped_column(
            Vector(EMBEDDING_DIMENSION),
            nullable=True,
            default=None,
        )
    # Store embeddings as JSON arrays (works across SQLite/Postgres without pgvector)
    return mapped_column(JSON, nullable=True, default=None)


def cosine_similarity(vec1: Sequence[float] | None, vec2: Sequence[float] | None) -> float | None:
    if not vec1 or not vec2:
        return None
    if len(vec1) != len(vec2):
        return None
    dot = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = sqrt(sum(a * a for a in vec1))
    norm2 = sqrt(sum(b * b for b in vec2))
    if norm1 == 0 or norm2 == 0:
        return None
    return dot / (norm1 * norm2)


class Base(DeclarativeBase, MappedAsDataclass):
    pass


class BaseObject(Base):
    __abstract__ = True


class User(Base):
    __tablename__ = "user"
    

    user_id: Mapped[str] = mapped_column(String(36), primary_key=True, init=False, default=lambda: str(uuid4()))
    username: Mapped[str] = mapped_column(String(128))
    name: Mapped[str] = mapped_column(String(256))
    role: Mapped[str | None] = mapped_column(String(128), nullable=True)
    profile_image: Mapped[str | None] = mapped_column(String, nullable=True)
    verification_token: Mapped[str | None] = mapped_column(String, nullable=True)
    reset_token: Mapped[str | None] = mapped_column(String, nullable=True)
    token_expiration: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    datetime_registered: Mapped[datetime]
    status_change_date: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    password_hash: Mapped[str]
    password_salt: Mapped[str]

    status: Mapped[str] = mapped_column(String(16), default="inactive")
    include_own_documents_in_feedback: Mapped[bool] = mapped_column(Boolean, default=False)
    default_publish: Mapped[bool] = mapped_column(Boolean, default=True)
    preferred_feedback_length: Mapped[str] = mapped_column(String(16), default="Brief")
    hide_affiliations: Mapped[bool] = mapped_column(Boolean, default=False)

    groups = relationship("Group", secondary="user_to_group", back_populates="users")
    documents = relationship(
        "Document",
        back_populates="user",
        cascade="all, delete",
        passive_deletes=True,
    )
    notes = relationship(
        "Note",
        back_populates="author",
        cascade="all, delete",
        passive_deletes=True,
    )

    activities = relationship(
        "Activity",
        back_populates="user",
        cascade="all, delete-orphan"
    )

    def __init__(
        self,
        username: str,
        name: str,
        datetime_registered: datetime,
        verification_token: str | None,
        token_expiration: datetime | None,
    ):
        super().__init__()
        self.username = username
        self.name = name
        self.datetime_registered = datetime_registered
        self.verification_token = verification_token
        self.token_expiration = token_expiration
        self.status = "inactive"
        self.status_change_date = datetime.now(timezone.utc)

    def set_password(self, password: str) -> str:
        salt = os.urandom(64)
        self.password_salt = binascii.hexlify(salt).decode("utf-8")
        hash_bytes = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 100000)
        self.password_hash = binascii.hexlify(hash_bytes).decode("utf-8")
        return self.password_hash

    def check_password(self, password: str) -> bool:
        if not self.password_salt or not self.password_hash:
            return False
        salt = binascii.unhexlify(self.password_salt.encode("utf-8"))
        hash_bytes = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 100000)
        hash_hex = binascii.hexlify(hash_bytes).decode("utf-8")
        return secrets.compare_digest(self.password_hash, hash_hex)


class Session(Base):
    __tablename__ = "session"
    

    id: Mapped[str] = mapped_column(String(128), primary_key=True, init=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("user.user_id", ondelete="CASCADE"), index=True)
    datetime_created: Mapped[datetime]
    datetime_valid_until: Mapped[datetime]


class Group(BaseObject):
    __tablename__ = "group"
    

    group_id: Mapped[str] = mapped_column(String(36), primary_key=True, init=False, default=lambda: str(uuid4()))
    name: Mapped[str] = mapped_column(String(256))
    datetime_created: Mapped[datetime]
    group_image: Mapped[str | None] = mapped_column(String, nullable=True)
    category: Mapped[str] = mapped_column(String(256), default="Other")
    description: Mapped[str] = mapped_column(Text, default="")
    visibility: Mapped[str] = mapped_column(String(32), default="public")

    users = relationship("User", secondary="user_to_group", back_populates="groups")
    admins = relationship("Administrator", secondary="admin_to_group", back_populates="groups")
    documents = relationship("Document", secondary="document_to_group", back_populates="groups")
    notes = relationship("Note", secondary="note_to_group", back_populates="groups")

    activities = relationship(
        "Activity",
        back_populates="group",
        cascade="all, delete-orphan"
    )

    __mapper_args__ = {"primary_key": [group_id]}

    @property
    def document_count(self) -> int:
        return len(self.documents)

    @property
    def user_count(self) -> int:
        return len(self.users)

    @property
    def first_three_user_profile_images(self) -> list[str | None]:
        return [user.profile_image for user in self.users[:3]]


class Administrator(Base):
    __tablename__ = "administrator"
    

    admin_id: Mapped[str] = mapped_column(String(36), primary_key=True, init=False, default=lambda: str(uuid4()))
    user_id: Mapped[str] = mapped_column(ForeignKey("user.user_id", ondelete="CASCADE"), index=True)

    user = relationship("User")
    groups = relationship("Group", secondary="admin_to_group", back_populates="admins")


class JoinRequest(Base):
    __tablename__ = "join_request"
    

    request_id: Mapped[int] = mapped_column(Identity(), primary_key=True, autoincrement=True, init=False)
    user_id: Mapped[str] = mapped_column(ForeignKey("user.user_id", ondelete="CASCADE"))
    group_id: Mapped[str] = mapped_column(ForeignKey("group.group_id", ondelete="CASCADE"))
    datetime_requested: Mapped[datetime] = mapped_column(default=datetime.now(timezone.utc), init=False)

    user = relationship("User")
    group = relationship("Group")


class GroupInvitation(Base):
    __tablename__ = "group_invitation"
    

    invitation_id: Mapped[int] = mapped_column(Integer, Identity(), primary_key=True, autoincrement=True, init=False)
    group_id: Mapped[str] = mapped_column(ForeignKey("group.group_id", ondelete="CASCADE"))
    inviter_id: Mapped[str] = mapped_column(ForeignKey("user.user_id", ondelete="CASCADE"))
    token: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    email: Mapped[str | None] = mapped_column(String(256), nullable=True)
    datetime_expiration: Mapped[datetime]
    datetime_created: Mapped[datetime] = mapped_column(default=datetime.now(timezone.utc), init=False)
    status: Mapped[str] = mapped_column(String(32), default="pending")

    group = relationship("Group")
    inviter = relationship("User")


class Document(BaseObject):
    __tablename__ = "document"
    

    document_id: Mapped[str] = mapped_column(String(36), primary_key=True, init=False, default=lambda: str(uuid4()))
    user_id: Mapped[str] = mapped_column(ForeignKey("user.user_id", ondelete="CASCADE"), index=True)
    author_id: Mapped[str]
    title: Mapped[str]
    content: Mapped[str] = mapped_column(Text)
    doc_type: Mapped[str]
    datetime_created: Mapped[datetime]
    datetime_modified: Mapped[datetime]
    embedding: Mapped[list[float] | None] = _embedding_column()
    file_key: Mapped[str | None] = mapped_column(String, nullable=True, default=None)
    image_key: Mapped[str | None] = mapped_column(String, nullable=True, default=None)
    is_published: Mapped[bool] = mapped_column(Boolean, default=False)

    user = relationship("User", back_populates="documents")
    groups = relationship("Group", secondary="document_to_group", back_populates="documents")
    chunks = relationship(
        "Chunk",
        back_populates="document",
        cascade="all, delete",
        passive_deletes=True,
    )
    references = relationship(
        "Reference",
        back_populates="document",
        cascade="all, delete",
        passive_deletes=True,
    )
    notes = relationship(
        "Note",
        back_populates="document",
        cascade="all, delete",
        passive_deletes=True,
    )


class Note(Base):
    __tablename__ = "note"
    

    note_id: Mapped[str] = mapped_column(String(36), primary_key=True, init=False, default=lambda: str(uuid4()))
    document_id: Mapped[str] = mapped_column(ForeignKey("document.document_id", ondelete="CASCADE"), index=True)
    author_id: Mapped[str] = mapped_column(ForeignKey("user.user_id", ondelete="CASCADE"), index=True)
    group_id: Mapped[str | None] = mapped_column(ForeignKey("group.group_id", ondelete="CASCADE"), index=True, nullable=True)
    content: Mapped[str] = mapped_column(Text)
    embedding: Mapped[list[float] | None] = _embedding_column()
    datetime_created: Mapped[datetime] = mapped_column(default=datetime.now(timezone.utc))

    document = relationship("Document", back_populates="notes")
    author = relationship("User", back_populates="notes")
    groups = relationship("Group", back_populates="notes")
    chunks = relationship(
        "Chunk",
        back_populates="note",
        cascade="all, delete",
        passive_deletes=True,
    )
    references = relationship(
        "Reference",
        back_populates="note",
        cascade="all, delete",
        passive_deletes=True,
    )


class Chunk(Base):
    __tablename__ = "chunk"
    

    chunk_id: Mapped[str] = mapped_column(String(36), primary_key=True, init=False, default=lambda: str(uuid4()))
    hash: Mapped[str] = mapped_column(String(32))
    content: Mapped[str] = mapped_column(Text)
    document_id: Mapped[str | None] = mapped_column(ForeignKey("document.document_id", ondelete="CASCADE"), index=True, nullable=True)
    note_id: Mapped[str | None] = mapped_column(ForeignKey("note.note_id", ondelete="CASCADE"), index=True, nullable=True)
    chunk_type: Mapped[str] = mapped_column(String(16), default="document")
    embedding: Mapped[list[float] | None] = _embedding_column()

    document = relationship("Document", back_populates="chunks")
    note = relationship("Note", back_populates="chunks")
    references = relationship(
        "Reference",
        back_populates="chunk",
        cascade="all, delete",
        passive_deletes=True,
    )
    feedbacks = relationship(
        "Feedback",
        back_populates="chunk",
        cascade="all, delete",
        passive_deletes=True,
    )


class Reference(Base):
    __tablename__ = "reference"
    

    reference_id: Mapped[str] = mapped_column(String(36), primary_key=True, init=False, default=lambda: str(uuid4()))
    source_chunk_id: Mapped[str] = mapped_column(ForeignKey("chunk.chunk_id", ondelete="CASCADE"), index=True)
    reference_document_id: Mapped[str | None] = mapped_column(ForeignKey("document.document_id", ondelete="CASCADE"), index=True, nullable=True)
    reference_note_id: Mapped[str | None] = mapped_column(ForeignKey("note.note_id", ondelete="CASCADE"), index=True, nullable=True)
    reference_type: Mapped[str] = mapped_column(String(16), default="document")

    chunk = relationship("Chunk", back_populates="references")
    document = relationship("Document", back_populates="references")
    note = relationship("Note", back_populates="references")


class Feedback(Base):
    __tablename__ = "feedback"


    feedback_id: Mapped[str] = mapped_column(String(36), primary_key=True, init=False, default=lambda: str(uuid4()))
    source_chunk_id: Mapped[str] = mapped_column(ForeignKey("chunk.chunk_id", ondelete="CASCADE"), index=True)
    feedback: Mapped[str] = mapped_column(Text)
    model: Mapped[str] = mapped_column(Text)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.now(timezone.utc))
    user_feedback: Mapped[str | None] = mapped_column(String(16), nullable=True, default=None)
    is_hidden: Mapped[bool] = mapped_column(Boolean, default=False)

    chunk = relationship("Chunk", back_populates="feedbacks")


class Activity(Base):
    __tablename__ = "activity"

    activity_id: Mapped[int] = mapped_column(Identity(), primary_key=True, init=False, autoincrement=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("user.user_id", ondelete="CASCADE"), nullable=False)
    group_id: Mapped[str] = mapped_column(ForeignKey("group.group_id", ondelete="CASCADE"), nullable=False)
    action: Mapped[str] = mapped_column(String(32))
    object_id: Mapped[str] = mapped_column(String(36))
    object_name: Mapped[str] = mapped_column(Text)
    object_type: Mapped[str] = mapped_column(String(32))
    timestamp: Mapped[datetime] = mapped_column(default=datetime.now(timezone.utc))

    user = relationship("User", back_populates="activities", lazy="joined")
    group = relationship("Group", back_populates="activities")


user_to_group_table = Table(
    "user_to_group",
    Base.metadata,
    Column("user_id", ForeignKey("user.user_id", ondelete="CASCADE"), primary_key=True),
    Column("group_id", ForeignKey("group.group_id", ondelete="CASCADE"), primary_key=True),
)


admin_to_group_table = Table(
    "admin_to_group",
    Base.metadata,
    Column("admin_id", ForeignKey("administrator.admin_id", ondelete="CASCADE"), primary_key=True),
    Column("group_id", ForeignKey("group.group_id", ondelete="CASCADE"), primary_key=True),
)


document_to_group_table = Table(
    "document_to_group",
    Base.metadata,
    Column("document_id", ForeignKey("document.document_id", ondelete="CASCADE"), primary_key=True),
    Column("group_id", ForeignKey("group.group_id", ondelete="CASCADE"), primary_key=True),
)

note_to_group_table = Table(
    "note_to_group",
    Base.metadata,
    Column("note_id", ForeignKey("note.note_id", ondelete="CASCADE"), primary_key=True),
    Column("group_id", ForeignKey("group.group_id", ondelete="CASCADE"), primary_key=True),
)


try:
    from compair_cloud.models import extend_models  # type: ignore
except (ImportError, ModuleNotFoundError):
    extend_models = None

if extend_models:
    extend_models(Base, globals())
