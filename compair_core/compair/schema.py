from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel


class GroupForm:
    name: str
    user_id: Optional[str] = None
    group_id: Optional[str] = None
    datetime_created: Optional[datetime] = None
    category: Optional[str] = None
    description: Optional[str] = None
    visibility: Optional[str] = None


class Group(BaseModel):
    name: str
    user_id: Optional[str] = None
    group_id: Optional[str] = None
    datetime_created: Optional[datetime] = None
    group_image: Optional[str] = None
    category: Optional[str] = None
    description: Optional[str] = None
    visibility: Optional[str] = None

    model_config = {"from_attributes": True}


class User(BaseModel):
    user_id: str
    username: str
    name: str
    datetime_registered: datetime
    status: str
    groups: Optional[list[Group]] = None
    profile_image: Optional[str] = None
    role: Optional[str] = None
    include_own_documents_in_feedback: Optional[bool] = None
    preferred_feedback_length: Optional[str] = None

    model_config = {"from_attributes": True}


class UpdateUserRequest(BaseModel):
    user_id: str
    name: Optional[str] = None
    role: Optional[str] = None
    group_ids: Optional[list[str]] = None


class Session(BaseModel):
    id: str
    user_id: str
    datetime_created: datetime
    datetime_valid_until: datetime


class Document(BaseModel):
    document_id: str
    user_id: str
    author_id: str
    groups: list[Group]
    user: User
    title: str
    content: str
    doc_type: str
    datetime_created: datetime
    datetime_modified: datetime
    is_published: bool
    file_key: Optional[str] = None
    image_key: Optional[str] = None
    topic_tags: Optional[list[str]] = None

    model_config = {"from_attributes": True}


class Chunk(BaseModel):
    chunk_id: str
    hash: str
    document_id: str
    content: str


class Reference(BaseModel):
    reference_id: str
    source_chunk_id: str
    reference_document_id: str | None = None
    reference_note_id: str | None = None
    reference_type: str = "document"
    reference_chunk_id: str | None = None
    document: Document | None = None
    document_author: str | None = None
    title: str | None = None
    author: str | None = None
    content: str | None = None
    file_path: str | None = None


class Feedback(BaseModel):
    feedback_id: str
    source_chunk_id: str
    feedback: str
    user_feedback: str | None = None


class LoginRequest(BaseModel):
    username: str
    password: str


class SignUpRequest(BaseModel):
    username: str
    name: str
    password: str
    groups: list[Group] | None
    referral_code: str | None = None


class ForgotPasswordRequest(BaseModel):
    email: str


class ResetPasswordRequest(BaseModel):
    token: str
    new_password: str


class Note(BaseModel):
    note_id: str
    document_id: str
    author_id: str
    group_id: str | None = None
    content: str
    datetime_created: datetime
    author: User | None = None
    group: Group | None = None


class InviteToGroupRequest(BaseModel):
    admin_id: str
    group_id: str
    email: str


class InviteMemberRequest(BaseModel):
    admin_id: str
    group_id: str
    username: str


class RemoveMemberRequest(BaseModel):
    group_id: str
    user_id: str


class LeaveGroupRequest(BaseModel):
    group_id: str


class NowReviewRequest(BaseModel):
    group_id: str
    document_ids: list[str] | None = None
    max_findings: int = 12
    model: str | None = None


class NowReviewResponse(BaseModel):
    group_id: str
    group_name: str
    document_ids: list[str]
    markdown: str
    findings: list[dict[str, Any]]
    meta: dict[str, Any]
