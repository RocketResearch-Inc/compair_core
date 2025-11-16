import hashlib
import os
import re
import requests
import secrets
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping, Optional, Tuple

import httpx
import psutil
from celery.result import AsyncResult
from fastapi import APIRouter, Body, Depends, File, Form, Header, HTTPException, Query, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.routing import APIRoute
from sqlalchemy import distinct, func, select, or_
from sqlalchemy.orm import joinedload, Session

from .server.deps import get_analytics, get_billing, get_ocr, get_settings_dependency, get_storage
from .server.providers.contracts import Analytics, BillingProvider, OCRProvider, StorageProvider
from .server.settings import Settings

from . import compair
from .compair import models, schema
from .compair.embeddings import create_embedding, Embedder
from .compair.logger import log_event 
from .compair.utils import chunk_text, generate_verification_token, log_activity
from .compair_email.email import emailer, EMAIL_USER
from .compair_email.templates import (
    ACCOUNT_VERIFY_TEMPLATE, 
    GROUP_INVITATION_TEMPLATE, 
    GROUP_JOIN_TEMPLATE, 
    INDIVIDUAL_INVITATION_TEMPLATE, 
    PASSWORD_RESET_TEMPLATE, 
    REFERRAL_CREDIT_TEMPLATE
)
from .compair.tasks import process_document_task as process_document_celery, send_feature_announcement_task, send_deactivate_request_email, send_help_request_email

try:
    import redis  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    redis = None


def _getenv(*names: str, default: Optional[str] = None) -> Optional[str]:
    """Return the first populated environment variable in the provided list."""
    for name in names:
        if not name:
            continue
        value = os.getenv(name)
        if value:
            return value
    return default

redis_url = _getenv("COMPAIR_REDIS_URL", "REDIS_URL")
redis_client = redis.Redis.from_url(redis_url) if (redis and redis_url) else None
#from compair.main import process_document

router = APIRouter()
core_router = APIRouter()
WEB_URL = os.environ.get("WEB_URL")
ADMIN_API_KEY = os.environ.get("ADMIN_API_KEY")

CLOUDFLARE_IMAGES_ACCOUNT = os.environ.get("CLOUDFLARE_IMAGES_ACCOUNT")
CLOUDFLARE_IMAGES_URL_ACCOUNT = os.environ.get("CLOUDFLARE_IMAGES_URL_ACCOUNT")
CLOUDFLARE_IMAGES_BASE_URL = f"https://imagedelivery.net/{CLOUDFLARE_IMAGES_URL_ACCOUNT}"
CLOUDFLARE_IMAGES_TOKEN = os.environ.get("CLOUDFLARE_IMAGES_TOKEN")
CLOUDFLARE_IMAGES_UPLOAD_URL = f"https://api.cloudflare.com/client/v4/accounts/{CLOUDFLARE_IMAGES_ACCOUNT}/images/v1"

GA4_MEASUREMENT_ID = _getenv("COMPAIR_GA4_MEASUREMENT_ID", "GA4_MEASUREMENT_ID")
GA4_API_SECRET = _getenv("COMPAIR_GA4_API_SECRET", "GA4_API_SECRET")

IS_CLOUD = os.getenv("COMPAIR_EDITION", "core").lower() == "cloud"
SINGLE_USER_SESSION_TTL = timedelta(days=365)


def _render_email(template: str, **context: str) -> str:
    """Lightweight template renderer for {{placeholders}} found in email HTML."""
    rendered = template
    for key, value in context.items():
        replacement = value or ""
        rendered = rendered.replace(f"{{{{{key}}}}}", replacement)
    return rendered


def _dispatch_process_document_task(
    user_id: str,
    doc_id: str,
    doc_text: str,
    generate_feedback: bool,
):
    task_callable = getattr(process_document_celery, "delay", None)
    if callable(task_callable):
        return task_callable(user_id, doc_id, doc_text, generate_feedback)
    return process_document_celery(user_id, doc_id, doc_text, generate_feedback)


def _ensure_single_user(session: Session, settings: Settings) -> models.User:
    """Create or fetch the singleton user used when authentication is disabled."""
    changed = False
    user = (
        session.query(models.User)
        .options(joinedload(models.User.groups))
        .filter(models.User.username == settings.single_user_username)
        .first()
    )
    if user is None:
        now = datetime.now(timezone.utc)
        user = models.User(
            username=settings.single_user_username,
            name=settings.single_user_name,
            datetime_registered=now,
            verification_token=None,
            token_expiration=None,
        )
        user.set_password(secrets.token_urlsafe(16))
        user.status = "active"
        user.status_change_date = now
        session.add(user)
        session.flush()
        admin = models.Administrator(user_id=user.user_id)
        group = models.Group(
            name=user.username,
            datetime_created=now,
            group_image=None,
            category="Private",
            description=f"Private workspace for {settings.single_user_name}",
            visibility="private",
        )
        group.admins.append(admin)
        user.groups = [group]
        session.add_all([group, admin])
        changed = True
    else:
        now = datetime.now(timezone.utc)
        if user.status != "active":
            user.status = "active"
            user.status_change_date = now
            changed = True
        group = next((g for g in user.groups if g.name == user.username), None)
        if group is None:
            group = session.query(models.Group).filter(models.Group.name == user.username).first()
            if group is None:
                group = models.Group(
                    name=user.username,
                    datetime_created=now,
                    group_image=None,
                    category="Private",
                    description=f"Private workspace for {user.name}",
                    visibility="private",
                )
                session.add(group)
                changed = True
            if group not in user.groups:
                user.groups.append(group)
                changed = True
        admin = session.query(models.Administrator).filter(models.Administrator.user_id == user.user_id).first()
        if admin is None:
            admin = models.Administrator(user_id=user.user_id)
            session.add(admin)
            changed = True
        if admin not in group.admins:
            group.admins.append(admin)
            changed = True

    if changed:
        session.commit()
        user = (
            session.query(models.User)
            .options(joinedload(models.User.groups))
            .filter(models.User.username == settings.single_user_username)
            .first()
        )
    if user is None:
        raise RuntimeError("Failed to initialize the local Compair user.")
    user.groups  # ensure relationship is loaded before detaching
    return user


def _ensure_single_user_session(session: Session, user: models.User) -> models.Session:
    """Return a long-lived session token for the singleton user."""
    now = datetime.now(timezone.utc)
    existing = (
        session.query(models.Session)
        .filter(models.Session.user_id == user.user_id, models.Session.datetime_valid_until >= now)
        .order_by(models.Session.datetime_valid_until.desc())
        .first()
    )
    if existing:
        return existing
    token = secrets.token_urlsafe()
    user_session = models.Session(
        id=token,
        user_id=user.user_id,
        datetime_created=now,
        datetime_valid_until=now + SINGLE_USER_SESSION_TTL,
    )
    session.add(user_session)
    session.commit()
    return user_session


def require_cloud(feature: str) -> None:
    if not IS_CLOUD:
        raise HTTPException(status_code=501, detail=f"{feature} is only available in the Compair Cloud edition.")


def _user_plan(user: models.User) -> str:
    return getattr(user, "plan", "free") or "free"


def _user_team(user: models.User):
    return getattr(user, "team", None)


def _trial_expiration(user: models.User) -> datetime | None:
    return getattr(user, "trial_expiration_date", None)


HAS_TEAM = hasattr(models, "Team")
HAS_ACTIVITY = hasattr(models, "Activity")
HAS_REFERRALS = hasattr(models.User, "referral_code")
HAS_BILLING = hasattr(models.User, "stripe_customer_id")
HAS_TRIALS = hasattr(models.User, "trial_expiration_date")
HAS_REDIS = redis_client is not None


def require_feature(flag: bool, feature: str) -> None:
    if not flag and not IS_CLOUD:
        raise HTTPException(status_code=501, detail=f"{feature} is only available in the Compair Cloud edition.")

def get_current_user(auth_token: str | None = Header(None)):
    settings = get_settings_dependency()
    if not settings.require_authentication:
        with compair.Session() as session:
            return _ensure_single_user(session, settings)
    if not auth_token:
        raise HTTPException(status_code=401, detail="Missing session token")
    with compair.Session() as session:
        user_session = session.query(models.Session).filter(models.Session.id == auth_token).first()
        if not user_session:
            raise HTTPException(status_code=401, detail="Invalid or expired session token")
        # Ensure datetime_valid_until is timezone-aware
        valid_until = user_session.datetime_valid_until
        if valid_until.tzinfo is None:
            valid_until = valid_until.replace(tzinfo=timezone.utc)
        if valid_until < datetime.now(timezone.utc):
            raise HTTPException(status_code=401, detail="Invalid or expired session token")
        user = session.query(
            models.User
        ).filter(
            models.User.user_id == user_session.user_id
        ).options(
            joinedload(models.User.groups)
        ).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        return user

def get_current_user_with_access_to_doc(
    document_id: str,
    current_user: models.User = Depends(get_current_user)
) -> models.User:
    with compair.Session() as session:
        doc = session.query(models.Document).filter(models.Document.document_id == document_id).first()
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")
        # Allow if user is author
        if doc.author_id == current_user.user_id:
            return current_user
        # Allow if user is in any group that has access to the document
        doc_group_ids = {g.group_id for g in doc.groups}
        user_group_ids = {g.group_id for g in current_user.groups}
        if doc_group_ids & user_group_ids:
            return current_user
        # Optionally, allow if document is published
        if doc.is_published:
            return current_user
        raise HTTPException(status_code=403, detail="Not authorized to access this document")

def log_service_resource_metrics(service_name="backend"):
    def log():
        try:
            p = psutil.Process(os.getpid())
            mem_mb = round(p.memory_info().rss / 1024 / 1024, 2)
            cpu_percent = p.cpu_percent(interval=1)
            log_event("service_resource", service=service_name, memory_mb=mem_mb, cpu_percent=cpu_percent)
        except Exception as e:
            print(f"[Resource Log Error] {e}")
        finally:
            # Re-schedule logging in 5 minutes
            threading.Timer(300, log).start()
    log()

log_service_resource_metrics(service_name="backend")  # or "frontend"

# Run via: fastapi dev api.py


@router.post("/login")
def login(request: schema.LoginRequest) -> dict:
    settings = get_settings_dependency()
    with compair.Session() as session:
        if not settings.require_authentication:
            user = _ensure_single_user(session, settings)
            user_session = _ensure_single_user_session(session, user)
            return {
                "user_id": user.user_id,
                "username": user.username,
                "name": user.name,
                "status": user.status,
                "role": user.role,
                "auth_token": user_session.id,
            }
        user = session.query(models.User).filter(models.User.username == request.username).first()
        if not user or not user.check_password(request.password):
            raise HTTPException(status_code=401, detail="Invalid credentials")
        if user.status == 'inactive':
            raise HTTPException(status_code=403, detail="User account is not verified")
        now = datetime.now(tz=timezone.utc)
        user_session = models.Session(
            id=secrets.token_urlsafe(),
            user_id=user.user_id,
            datetime_created=now,
            datetime_valid_until=now + timedelta(days=1),
        )
        session.add(user_session)
        session.commit()
        return {
            "user_id": user.user_id,
            "username": user.username,
            "name": user.name,
            "status": user.status,
            "role": user.role,
            "auth_token": user_session.id,  # Return the session token here
        }


@router.get("/username_exists")
def username_exists(username: str) -> dict:
    with compair.Session() as session:
        exists = session.query(models.User).filter(models.User.username == username).first() is not None
        return {"exists": exists}


@router.get("/load_user")
def load_user(
    username: str,
) -> schema.User | None:
    with compair.Session() as session:
        q = select(models.User).filter(
            models.User.username.match(username)
        )
        user = session.execute(q).fetchone()
        if user is None:
            return
        user = user[0]
        if user.groups is None:
            user.groups = []
        print(f'User: {user}')
        return user


@router.get("/load_user_plan")
def load_user(
    current_user: models.User = Depends(get_current_user)
) -> dict:
    return {'plan': _user_plan(current_user)}


@router.get("/load_user_by_id")
def load_user(
    user_id: str,
) -> schema.User | None:
    with compair.Session() as session:
        q = select(models.User).filter(
            models.User.user_id==user_id
        )
        user = session.execute(q).fetchone()
        if user is None:
            return
        user = user[0]
        if user.groups is None:
            user.groups = []
        print(f'User: {user}')
        return user


@router.get("/load_user_files")
def load_user_files(
    connection_id: str,
    page: int = 1,
    page_size: int = 10,
    filter_type: str | None = None,
    current_user: models.User = Depends(get_current_user)
) -> dict:
    with compair.Session() as session:
        # Validate connection
        connection = session.query(models.User).filter(models.User.user_id == connection_id).first()
        if not connection:
            return {"files": [], "message": "User or connection not found."}

        # Security: must share a group
        shared_group_ids = set(g.group_id for g in current_user.groups).intersection(g.group_id for g in connection.groups)
        if not shared_group_ids:
            return {"files": [], "message": "You do not have permission to view this user's affiliations, as you do not share a group together."}

        # Privacy: check setting
        if connection.hide_affiliations:
            return {"files": [], "message": f"{connection.username} has set their profile to private."}

        now = datetime.now(timezone.utc)
        week_ago = now - timedelta(days=7)
        # Fetch documents belonging to shared or public groups
        q = select(models.Document).join(models.Document.groups).filter(
            models.Document.user_id == connection_id,
            models.Group.group_id.in_(shared_group_ids) |  # Shared groups
            (
                (models.Group.visibility == "public") &  # Public groups
                models.Group.users.any(models.User.user_id == connection_id)  # Associated with the connection
            )
        ).options(
            joinedload(models.Document.groups),
            joinedload(models.Document.user).joinedload(models.User.groups)
        )

        # --- Filter logic ---
        if filter_type == "published":
            q = q.filter(models.Document.is_published == True)
        elif filter_type == "unpublished":
            q = q.filter(models.Document.is_published == False)
        elif filter_type == "recently_updated":
            # Documents updated OR with a note in the past week
            q = q.outerjoin(models.Document.notes).filter(
                or_(
                    models.Document.datetime_modified >= week_ago,
                    models.Note.datetime_created >= week_ago
                )
            )
        elif filter_type == "recently_compaired":
            # Documents with feedback in the past week
            q = q.join(models.Document.chunks).join(models.Chunk.feedbacks).filter(
                models.Feedback.timestamp >= week_ago
            )
        # Default: all, sorted by last update
        q = q.order_by(models.Document.datetime_modified.desc())

        documents = session.execute(q).unique().fetchall()

        if documents is None or len(documents)==0:
            return {
                "documents": [],
                "total_count": 0,
                "message": None
            }

        total_count = len(documents)
        # Paging
        offset = (page - 1) * page_size
        documents = session.execute(q.order_by(models.Document.datetime_created.desc()).offset(offset).limit(page_size)).unique().fetchall()

        files = [d[0] for d in documents] if documents else []
        return {
            "files": [schema.Document.model_validate(f) for f in files], 
            "message": None,
            "total_count": total_count,
        }


@router.get("/load_user_groups")
def load_user_groups(
    connection_id: str,
    page: int = 1,
    page_size: int = 10,
    filter_type: str | None = None,
    current_user: models.User = Depends(get_current_user)
) -> dict:
    with compair.Session() as session:
        now = datetime.now(timezone.utc)
        week_ago = now - timedelta(days=7)
        # Validate connection
        connection = session.query(models.User).filter(models.User.user_id == connection_id).first()
        if not connection:
            return {"groups": [], "message": "User or connection not found."}

        # Check if the connection is valid
        shared_group_ids = set([g.group_id for g in current_user.groups]).intersection([g.group_id for g in connection.groups])
        if not shared_group_ids:
            return {"groups": [], "message": "You do not have permission to view this user's affiliations, as you do not share a group together."}
        
        # Privacy: check setting
        if connection.hide_affiliations:
            return {"groups": [], "message": f"{connection.username} has set their profile to private."}

        # Fetch shared or public groups of the connection
        groups_query = session.query(models.Group).filter(
            models.Group.group_id.in_(shared_group_ids) |  # Shared groups
            (
                (models.Group.visibility == "public") &  # Public groups
                models.Group.users.any(models.User.user_id == connection_id)  # Associated with the connection
            )
        ).options(
            joinedload(models.Group.users),
            joinedload(models.Group.documents)
        )

        # --- Filter logic (match /load_groups) ---
        if filter_type == "internal":
            groups_query = groups_query.filter(models.Group.visibility == "internal")
        elif filter_type == "public":
            groups_query = groups_query.filter(models.Group.visibility == "public")
        elif filter_type == "private":
            groups_query = groups_query.filter(models.Group.visibility == "private")
        elif filter_type == "recently_updated":
            groups_query = groups_query.join(models.Group.documents).filter(
                models.Document.datetime_created >= week_ago
            )
        else:
            groups_query = groups_query.order_by(models.Group.name.asc())

        total_count = groups_query.count()
        offset = (page - 1) * page_size
        groups = groups_query.offset(offset).limit(page_size).all()

        result = [
            {
                "group_id": group.group_id,
                "name": group.name,
                "datetime_created": group.datetime_created,
                "group_image": group.group_image,
                "category": group.category,
                "description": group.description,
                "visibility": group.visibility,
                "document_count": getattr(group, "document_count", None),
                "user_count": getattr(group, "user_count", None),
                "first_three_user_profile_images": getattr(group, "first_three_user_profile_images", None)
            }
            for group in groups
        ]
        return {"groups": result, "message": None, "total_count": total_count}


@router.get("/load_user_status")
def load_user_status(
    current_user: models.User = Depends(get_current_user)
) -> str:
    user_status = 'inactive'
    with compair.Session() as session:
        user_status = current_user.status
        return user_status


@router.get("/load_user_status_date")
def load_user_status(
    current_user: models.User = Depends(get_current_user)
) -> datetime:
    if not (HAS_TRIALS or HAS_BILLING) and not IS_CLOUD:
        raise HTTPException(status_code=501, detail="User status dates are only tracked in the Compair Cloud edition.")
    with compair.Session() as session:
        user_status = current_user.status
        if user_status=='active':
            require_feature(HAS_BILLING, "Billing history")
            user_status_date = current_user.last_payment_date
        elif user_status=='trial':
            require_feature(HAS_TRIALS, "Trial management")
            user_status_date = current_user.trial_expiration_date
        elif user_status=='suspended':
            user_status_date = current_user.status_change_date
        else:
            raise HTTPException(status_code=403, detail='User Inactive')
        return user_status_date


@router.get("/load_referral_credits")
def load_referral_credits(
    current_user: models.User = Depends(get_current_user)
) -> Tuple[int, int]:
    if not HAS_REFERRALS and not IS_CLOUD:
        raise HTTPException(status_code=501, detail="Referral credits are only available in the Compair Cloud edition.")
    with compair.Session() as session:
        referral_credits_earned = current_user.referral_credits
        referral_credits_pending = current_user.pending_referral_credits
        return (referral_credits_earned, referral_credits_pending)


def create_user(
    username: str,
    name: str,
    password: str,
    session: Session,
    groups: list[str] | None = None,
    referral_code: str = None
):
    token, expiration = generate_verification_token()
    existing_user = session.query(models.User).filter(models.User.username == username).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already in use")

    user = models.User(
        username=username,
        name=name,
        datetime_registered=datetime.now(),
        verification_token=token.lower(),
        token_expiration=expiration,
    )
    user.set_password(password=password)
    session.add(user)
    session.commit()

    if HAS_TEAM:
        team_invitations = session.query(models.TeamInvitation).filter(
            models.TeamInvitation.email == username,
            models.TeamInvitation.status == "pending",
        ).all()

        for invitation in team_invitations:
            emailer.connect()
            emailer.send(
                subject="You're Invited to Join a Team on Compair",
                sender=EMAIL_USER,
                receivers=[invitation.email],
                html=f"""
                <p>{invitation.inviter.name} has invited you to join their team on Compair!</p>
                <p>Click <a href="https://{WEB_URL}/accept-invitation?token={invitation.invitation_id}">here</a> to join.</p>
                """
            )
            invitation.status = "sent"
            session.commit()

    if groups is not None:
        user.groups = load_groups_by_ids(groups)
    else:
        group = models.Group(
            name=username,
            datetime_created=datetime.now(),
            group_image=None,
            category="Private",
            description=f"A private group for {username}",
            visibility="private"
        )
        admin = models.Administrator(user_id=user.user_id)
        session.add(admin)
        session.add(group)
        session.commit()
        group.admins.append(admin)
        user.groups = [group]
    
    # Track referral if a code is provided
    if referral_code:
        require_feature(HAS_REFERRALS, "Referral program")
        print(f'Got to backend referral code: {referral_code}')
        referrer = session.query(models.User).filter(models.User.referral_code == referral_code).first()
        if referrer and hasattr(referrer, "referral_credits"):
            max_credits = 3
            if referrer.referral_credits <= max_credits * 10:  # $10 per credit
                if hasattr(user, "referred_by"):
                    user.referred_by = referrer.user_id  # Store who referred them
                if hasattr(referrer, "pending_referral_credits"):
                    referrer.pending_referral_credits += 10  # Add pending credit
                session.commit()
    
    session.add(user)
    session.commit()
    try:
        analytics.track("user_signup", user.user_id)
    except Exception as exc:
        print(f"analytics track failed: {exc}")
    return user


def _activate_user_account(
    session: Session,
    user: models.User,
    *,
    send_group_invites: bool = True,
) -> None:
    user.status = "trial" if HAS_TRIALS else "active"
    user.status_change_date = datetime.now(timezone.utc)
    if HAS_TRIALS:
        user.trial_expiration_date = datetime.now(timezone.utc) + timedelta(days=30)
    user.verification_token = None
    session.commit()

    if not send_group_invites:
        return

    pending_invitations = session.query(models.GroupInvitation).filter(
        models.GroupInvitation.email == user.username,
        models.GroupInvitation.status == "pending"
    ).all()
    for invitation in pending_invitations:
        invitation.status = "sent"
        if not invitation.token:
            invitation.token = secrets.token_urlsafe(32).lower()
        invitation.datetime_expiration = datetime.now(timezone.utc) + timedelta(days=7)
        session.commit()

        group = invitation.group
        inviter = invitation.inviter
        invitation_link = f"http://{WEB_URL}/accept-group-invitation?token={invitation.token}&user_id={user.user_id}"
        emailer.connect()
        emailer.send(
            subject="You’re Invited to Join a Group on Compair",
            sender=EMAIL_USER,
            receivers=[user.username],
            html=GROUP_INVITATION_TEMPLATE.replace(
                "{{inviter_name}}", inviter.name
            ).replace(
                "{{group_name}}", group.name
            ).replace(
                "{{invitation_link}}", invitation_link
            )
        )


@router.get("/load_session")
def load_session(auth_token: str | None = None) -> schema.Session | None:
    settings = get_settings_dependency()
    if not settings.require_authentication:
        with compair.Session() as session:
            user = _ensure_single_user(session, settings)
            session_model = _ensure_single_user_session(session, user)
            return schema.Session.model_validate(session_model, from_attributes=True)
    with compair.Session() as session:
        if not auth_token:
            raise HTTPException(status_code=400, detail="auth_token is required when authentication is enabled.")
        user_session = session.query(models.Session).filter(models.Session.id == auth_token).first()
        if not user_session:
            raise HTTPException(status_code=404, detail="Session not found")
        valid_until = user_session.datetime_valid_until
        if valid_until.tzinfo is None:
            valid_until = valid_until.replace(tzinfo=timezone.utc)
        if valid_until < datetime.now(timezone.utc):
            raise HTTPException(status_code=401, detail="Invalid or expired session token")
        return schema.Session.model_validate(user_session, from_attributes=True)


@router.post("/update_user")
def update_user(
    name: str = Form(None),
    role: str = Form(None),
    group_ids: list[str] = Form(None),
    include_own_documents_in_feedback: str = Form(None),
    default_publish: str = Form(None),
    preferred_feedback_length: str = Form(None),
    hide_affiliations: str = Form(None),
    current_user: models.User = Depends(get_current_user)
):
    with compair.Session() as session:
        if name is not None:
            current_user.name = name
        if role is not None:
            current_user.role = role
        if group_ids is not None:
            groups = load_groups_by_ids(group_ids)
            groups = [g for g in groups if g not in current_user.groups]
            current_user.groups.extend(groups)
        if include_own_documents_in_feedback is not None:
            # Convert string to bool
            current_user.include_own_documents_in_feedback = include_own_documents_in_feedback.lower() == "true"
        if default_publish is not None:
            current_user.default_publish = default_publish.lower() == "true"
        if preferred_feedback_length is not None:
            # Lock to Brief for the time being
            preferred_feedback_length = 'Brief'
            current_user.preferred_feedback_length = preferred_feedback_length
        if hide_affiliations is not None:
            current_user.hide_affiliations = hide_affiliations.lower() == "true"
        session.add(current_user)
        session.commit()

@router.get("/update_session_duration")
def update_session_duration(
    user_session: schema.Session,
    new_valid_until: datetime,
) -> None:
    with compair.Session() as session:
        user_session = session.query(
            models.Session
        ).filter(
            models.Session.id == user_session.id
        ).first()
        user_session.update({'datetime_valid_until': new_valid_until})
        session.commit()


@router.get("/delete_user")
def delete_user(
    current_user: models.User = Depends(get_current_user)
):
    settings = get_settings_dependency()
    if not settings.require_authentication:
        raise HTTPException(status_code=403, detail="Deleting the local user is not supported when authentication is disabled.")
    with compair.Session() as session:
        current_user.delete()
        session.commit()


@router.get("/load_connections")
def load_connections(
    page: int = 1,
    page_size: int = 10,
    filter_type: str | None = None,
    current_user: models.User = Depends(get_current_user)
) -> dict | None:
    with compair.Session() as session:
        now = datetime.now(timezone.utc)
        week_ago = now - timedelta(days=7)

        # Get all groups the user belongs to
        groups = session.query(models.Group).options(joinedload(models.Group.users)).filter(
            models.Group.group_id.in_([g.group_id for g in current_user.groups])
        ).all()
        if not groups:
            return {"connections": [], "total_count": 0}

        # Collect all user IDs from the groups
        connection_ids = set()
        for group in groups:
            for group_user in group.users:
                if group_user.user_id != current_user.user_id:  # Exclude the requesting user
                    connection_ids.add(group_user.user_id)

        # Fetch the User objects for the collected IDs
        q = session.query(models.User).filter(models.User.user_id.in_(connection_ids))
        
        # --- Filter logic ---
        if filter_type == "recently_active":
            q = q.join(models.User.activities).filter(
                models.Activity.action == "create",
                models.Activity.timestamp >= week_ago
            )
        elif filter_type == "recently_compaired":
            # Their doc was used for feedback on your doc, or vice versa, in the past week
            q = q.join(models.User.documents).join(models.Document.chunks).join(models.Chunk.feedbacks).filter(
                models.Feedback.timestamp >= week_ago
            )
        else:
            q = q.order_by(models.User.name.asc())
        
        total_count = q.count()
        offset = (page - 1) * page_size
        connections = q.order_by(models.User.datetime_registered.desc()).offset(offset).limit(page_size).all()

        # Convert users to dictionary format
        return {
            "connections": [
                {
                    "user_id": connection.user_id,
                    "username": connection.username,
                    "name": connection.name,
                    "datetime_registered": connection.datetime_registered,
                    "status": connection.status,
                    "profile_image": connection.profile_image,
                    "role": connection.role,
                }
                for connection in connections
            ],
            "total_count": total_count
        }


@router.get("/all_group_categories")
def all_group_categories(
    current_user: models.User = Depends(get_current_user)
):
    """Return all unique group categories."""
    with compair.Session() as session:
        categories = (
            session.query(distinct(models.Group.category))
            .order_by(models.Group.category.asc())
            .all()
        )
        # Flatten list of tuples and filter out None/empty
        categories = [c[0] for c in categories if c[0] and c[0]!='Compair']
        return {"categories": categories}


@router.get("/load_groups")
def load_groups(
    user_id: str | None = None,
    page: int = 1,
    page_size: int = 10,
    filter_type: str | None = None,
    category: str | None = None,
    visibility: str | None = None,
    sort: str | None = None,
    query: str | None = None,
    own_groups_only: bool = False,
    current_user: models.User = Depends(get_current_user)
) -> dict | None:
    with compair.Session() as session:
        # --- User-based group selection ---
        if user_id is None:
            q = session.query(models.Group).options(
                joinedload(models.Group.users),
                joinedload(models.Group.documents)
            ).filter(
                models.Group.visibility != 'private'
            )
        else:
            user = session.query(models.User).options(
                joinedload(models.User.groups).joinedload(models.Group.users),
                joinedload(models.User.groups).joinedload(models.Group.documents)
            ).filter(
                models.User.user_id == current_user.user_id
            ).first()
            user_group_ids = [g.group_id for g in user.groups if g.category!='Compair']

            invited_group_ids = set()
            invitations = session.query(models.GroupInvitation).filter(
                models.GroupInvitation.email == user.username,
                models.GroupInvitation.status == "sent"
            ).all()
            invited_group_ids.update([i.group_id for i in invitations])

            accessible_group_ids = set(user_group_ids) | invited_group_ids

            if own_groups_only:
                # Only groups the user is a member of or invited
                q = session.query(models.Group).filter(
                    models.Group.group_id.in_(accessible_group_ids)
                )
            else:
                # All groups user can access: public, or private/internal if a member, or groups with an invitation
                q = session.query(models.Group).filter(
                    (models.Group.visibility == "public") |
                    (models.Group.group_id.in_(accessible_group_ids))
                )

        # --- Filtering ---
        if category and category.lower() != "all":
            q = q.filter(models.Group.category == category)
        if visibility and visibility.lower() != "all":
            q = q.filter(models.Group.visibility == visibility)
        if filter_type == "joined" and user_id:
            #user = session.query(models.User).filter(models.User.user_id == user_id).first()
            #q = q.filter(models.Group.group_id.in_([g.group_id for g in user.groups]))
            # Already constrained above if own_groups_only is True
            pass
        if filter_type == "pending" and user_id:
            # Groups with pending join requests or invitations for this user
            user = session.query(models.User).filter(models.User.user_id == current_user.user_id).first()
            pending_group_ids = set()
            invitations = session.query(models.GroupInvitation).filter(
                models.GroupInvitation.email == user.username,
                models.GroupInvitation.status == "sent"
            ).all()
            pending_group_ids.update([i.group_id for i in invitations])
            q = q.filter(models.Group.group_id.in_(pending_group_ids))
        if query:
            q = q.filter(models.Group.name.ilike(f"%{query}%"))
        # --- Sorting ---
        if sort == "popular":
            q = q.outerjoin(models.Group.users).group_by(models.Group.group_id).order_by(func.count(models.User.user_id).desc())
        elif sort == "recently_updated":
            q = q.outerjoin(models.Group.documents).group_by(models.Group.group_id).order_by(func.max(models.Document.datetime_modified).desc())
        elif sort == "recently_created":
            q = q.order_by(models.Group.datetime_created.desc())
        else:
            q = q.order_by(models.Group.name.asc())
        # --- Paging ---
        total_count = q.count()
        offset = (page - 1) * page_size
        groups = q.offset(offset).limit(page_size).all()
            
        result = [
            {
                "group_id": group.group_id,
                "name": group.name,
                "datetime_created": group.datetime_created,
                "group_image": group.group_image,
                "category": group.category,
                "description": group.description,
                "visibility": group.visibility,
                "document_count": group.document_count,
                "user_count": group.user_count,
                "first_three_user_profile_images": group.first_three_user_profile_images
            }
            for group in groups
        ]
        return {"groups": result, "total_count": total_count}


def load_groups_by_ids(group_ids: list[str]) -> list[schema.Group]:
    with compair.Session() as session:
        q = session.query(models.Group).filter(
            models.Group.group_id.in_(group_ids)
        )
        return q.all()  # Returns list of Group objects directly


@router.get("/load_group")
def load_group(
    name: str | None = None,
    group_id: str | None = None
) -> schema.Group | None:
    if (name is not None) or (group_id is not None):
        with compair.Session() as session:
            if group_id is not None:
                q = select(models.Group).filter(
                    models.Group.group_id==group_id
                )
            else:
                q = select(models.Group).filter(
                    models.Group.name.match(name)
                )
            group = session.execute(q).fetchone()
            if group is None:
                return None
            return group[0]


def notify_group_admins(
    group: models.Group, 
    user_id: str
):
    """Send an email notification to group admins."""
    with compair.Session() as session:
        admin_emails = [admin.user.username for admin in group.admins]
        if len(admin_emails) == 0:
            print("No admins found for group:", group.name)
            return
        user = session.query(models.User).filter(models.User.user_id == user_id).first()
        emailer.connect()
        print("Admin emails:", admin_emails)
        emailer.send(
            subject="Group Join Request",
            sender=EMAIL_USER,
            receivers=admin_emails,
            html=GROUP_JOIN_TEMPLATE.replace(
                "{{ user_name }}", user.username
            ).replace(
                "{{ group_name }}", group.name
            ).replace(
                "{{ admin_panel_url }}", f"http://{WEB_URL}/admin/groups"
            )
        )

@router.post("/join_group")
def join_group(
    group_id: str = Form(...),
    current_user: models.User = Depends(get_current_user)
):
    return join_group_direct(
        user_id=current_user.user_id, 
        group_id=group_id
    )

def join_group_direct(
    user_id: str, 
    group_id: str
):
    """Join a group based on its visibility."""
    print("1")
    with compair.Session() as session:
        group = session.query(models.Group).filter(models.Group.group_id == group_id).first()
        print(group)
        if not group:
            raise HTTPException(status_code=404, detail="Group not found")
        print(group.visibility)
        if group.visibility in ["public", "internal"]:
            user = session.query(models.User).filter(models.User.user_id == user_id).first()
            # Look for any existing invitations associated with this group
            invitations = session.query(models.GroupInvitation).filter(
                models.GroupInvitation.group_id == group_id,
                models.GroupInvitation.email == user.username,
                models.GroupInvitation.status == "sent"
            ).all()
            for invitation in invitations:
                invitation.status = "accepted"

            if group.visibility == "public":
                group.users.append(user)
                session.commit()
                log_activity(
                    session=session, 
                    user_id=user_id, 
                    group_id=group.group_id,
                    action="join", 
                    object_id=group.group_id, 
                    object_name=group.name,
                    object_type="group"
                )
                return {"message": "Joined group successfully"}
            
            elif group.visibility == "internal":
                if len(invitations)>0:
                     # Invitation found; add to group
                    group.users.append(user)
                    session.commit()
                else:
                    # Create a JoinRequest if not already present
                    existing_request = session.query(models.JoinRequest).filter(
                        models.JoinRequest.user_id == user_id,
                        models.JoinRequest.group_id == group_id
                    ).first()
                    if not existing_request:
                        join_request = models.JoinRequest(
                            user_id=user_id,
                            group_id=group_id,
                            datetime_requested=datetime.now(timezone.utc)
                        )
                        session.add(join_request)
                        session.commit()
                    notify_group_admins(group, user_id)
                return {"message": "Join request sent to group admins"}

        elif group.visibility == "private":
            raise HTTPException(status_code=403, detail="Cannot join private group without an invite")


@router.post("/create_group")
async def create_group(
    name: str = Form(...),
    category: str = Form(None),
    description: str = Form(None),
    visibility: str = Form("public"),
    file: UploadFile = File(None),  # Allow optional file upload
    current_user: models.User = Depends(get_current_user)
):
    with compair.Session() as session:
        print('1')
        if category not in all_group_categories()['categories']:
            category = "Other"  # Default to "Other" if category is not valid

        # Limit internal group creation to active, team plans
        if visibility == 'internal' and not (current_user.status == 'active' and _user_plan(current_user) == 'team'):
            raise HTTPException(
                status_code=403, 
                detail="Internal groups can only be created by users with an active team plan"
            )

        created_group = models.Group(
            name=name,
            group_image=None,
            category=category,
            description=description,
            visibility=visibility,
            datetime_created=datetime.now(),
        )
        print(created_group)
        # Check if user has an admin ID
        q = select(models.Administrator).filter(
            models.Administrator.user_id==current_user.user_id
        )
        admin = session.execute(q).fetchone()
        print(admin)
        if admin is None:
            # Make the user an admin
            admin = models.Administrator(user_id=current_user.user_id)
            session.add(admin)
            session.commit()
        else:
            admin = admin[0]
        print('3?')
        created_group.admins.append(admin)
        session.add(created_group)
        session.commit()

        # Log activity
        log_activity(
            session=session, 
            user_id=current_user.user_id, 
            group_id=created_group.group_id,
            action="create", 
            object_id=created_group.group_id, 
            object_name=created_group.name,
            object_type="group"
        )

        # Add group to user
        admin.user.groups.append(created_group)
        print('4??')
        session.add(admin)
        session.commit()

        if file is not None:
            await upload_group_image(
                group_id=created_group.group_id,
                upload_type='group',
                file=file
            )
        return {
            "group_id": created_group.group_id,
            "name": created_group.name,
            "visibility": created_group.visibility,
            "category": created_group.category,
        }


@router.get("/load_documents")
def load_doc(
    group_id: str | None = None,
    page: int = 1,
    page_size: int = 10,
    filter_type: str | None = None,
    own_documents_only: bool = True,
    current_user: models.User = Depends(get_current_user)
) -> Mapping[str, Any] | None:
    with compair.Session() as session:
        now = datetime.now(timezone.utc)
        week_ago = now - timedelta(days=7)

        # Get user and their group memberships
        user_group_ids = set(g.group_id for g in current_user.groups)

        q = session.query(models.Document)

        if own_documents_only:
            q = q.filter(models.Document.user_id == current_user.user_id)
        else:
            q = q.join(models.Document.groups)
            if group_id:
                q = q.filter(models.Group.group_id == group_id)
            q = q.filter(
                (models.Group.visibility == "public") |
                (
                    models.Group.group_id.in_(user_group_ids)
                )
            ).options(
                joinedload(models.Document.groups),
                joinedload(models.Document.user).joinedload(models.User.groups)
            )

        if group_id is not None and own_documents_only:
            q = q.filter(models.Document.groups.any(models.Group.group_id == group_id))

        # --- Filter logic: publishing ---
        if filter_type == "unpublished" and own_documents_only:
            q = q.filter(models.Document.is_published == False)
        else:
            q = q.filter(
                or_(
                    models.Document.is_published == True,
                    models.Document.user_id == current_user.user_id
                )
            )
        
        # --- Filter logic: other ---
        if filter_type == "recently_updated":
            # Documents updated OR with a note in the past week
            q = q.outerjoin(models.Document.notes).filter(
                or_(
                    models.Document.datetime_modified >= week_ago,
                    models.Note.datetime_created >= week_ago
                )
            )
        elif filter_type == "recently_compaired":
            # Documents with feedback in the past week
            q = q.join(models.Document.chunks).join(models.Chunk.feedbacks).filter(
                models.Feedback.timestamp >= week_ago
            )
        # Default: all, sorted by last update
        q = q.order_by(models.Document.datetime_modified.desc())

        documents = session.execute(q).unique().fetchall()
        print(documents)
        if documents is None or len(documents)==0:
            return {
                "documents": [],
                "total_count": 0
            }

        total_count = q.count()

        # Paging
        offset = (page - 1) * page_size
        documents = session.execute(q.order_by(models.Document.datetime_created.desc()).offset(offset).limit(page_size)).unique().fetchall()
        #print(documents)
        if documents is None or len(documents)==0:
            return {
                "documents": [],
                "total_count": 0
            }
        #print(f'API returning these documents: {documents}')
        print(f'Total count: {total_count}')
        print(f'Page: {page}, Page size: {page_size}, Offset: {offset}')
        return {
            "documents": [
                schema.Document.model_validate(d[0]) for d in documents
            ],
            "total_count": total_count
        }


@router.get("/load_group_users")
def load_group_users(
    group_id: str,
    page: int = 1,
    page_size: int = 10,
    filter_type: str | None = None,
    current_user: models.User = Depends(get_current_user)
) -> dict:
    print('1')
    print(current_user.user_id)
    print(group_id)
    with compair.Session() as session:
        # Check if the group exists
        group = session.query(models.Group).filter(models.Group.group_id == group_id).first()
        print(group)
        if not group:
            raise HTTPException(status_code=404, detail="Group not found")

        if (current_user not in group.users) & (group.visibility!='public'):
            raise HTTPException(status_code=403, detail="User does not belong to the group")

        # Retrieve all users associated with the group
        users_query = session.query(models.User).join(models.User.groups).filter(models.Group.group_id == group_id)

        # --- Filter logic ---
        if filter_type == "recently_active":
            users_query = users_query.order_by(models.User.status_change_date.desc())
        elif filter_type == "recently_joined":
            users_query = users_query.order_by(models.User.datetime_registered.desc())
        else:
            users_query = users_query.order_by(models.User.name.asc())

        total_count = users_query.count()
        offset = (page - 1) * page_size
        users = users_query.offset(offset).limit(page_size).all()

        # Convert users to schema objects
        return {
            "users": [
                {
                    "user_id": u.user_id,
                    "username": u.username,
                    "name": u.name,
                    "datetime_registered": u.datetime_registered,
                    "status": u.status,
                    "profile_image": u.profile_image,
                    "role": u.role,
                }
                for u in users
            ],
            "total_count": total_count
        }


@router.get("/load_document")
def load_doc(
    title: str,
    current_user: models.User = Depends(get_current_user)
) -> schema.Document | None:
    with compair.Session() as session:
        q = select(models.Document).filter(
            models.Document.user_id==current_user.user_id
        ).filter(
            models.Document.title.match(title)
        ).options(
            joinedload(models.Document.groups),
            joinedload(models.Document.user).joinedload(models.User.groups)
        )
        document = session.execute(q).unique().fetchone()
        if document is None:
            return None
        return document[0]


@router.get("/load_document_by_id")
def load_doc(
    document_id: str,
    current_user: models.User = Depends(get_current_user)
) -> schema.Document | None:
    with compair.Session() as session:
        q = select(models.Document).filter(
            models.Document.document_id==document_id
        ).options(
            joinedload(models.Document.groups),
            joinedload(models.Document.user).joinedload(models.User.groups)
        )
        document = session.execute(q).unique().fetchone()
        if document is None:
            return None
        doc = document[0]
        doc_group_ids = {g.group_id for g in doc.groups}
        user_group_ids = {g.group_id for g in current_user.groups}
        if not doc_group_ids & user_group_ids and current_user.user_id != doc.author_id:
            raise HTTPException(status_code=403, detail="Not authorized to view this document")
        return doc


@router.post("/update_doc")
def update_doc(
    doc_id: str = Form(...),
    author_id: str = Form(None),
    title: str = Form(None),
    datetime_created: datetime = Form(None),
    group_ids: list[str] = Form(None),
    image_url: str = Form(None),
    current_user: models.User = Depends(get_current_user)
):
    print('In update doc')
    print(author_id)
    print(title)
    print(datetime_created)
    print(group_ids)
    print(image_url)
    with compair.Session() as session:
        doc = session.query(models.Document).filter(
            models.Document.document_id == doc_id,
            models.Document.user_id == current_user.user_id
        ).first()
        if doc:
            if author_id is not None:
                doc.author_id = author_id
            if title is not None:
                doc.title = title
            if datetime_created is not None:
                doc.datetime_created = datetime_created
            if group_ids is not None:
                groups = load_groups_by_ids(group_ids)
                doc.groups = []
                doc.groups.extend(groups)
                print(f'New groups here? {doc.groups}')
            if image_url is not None:
                doc.image_url = image_url
            session.commit()


@router.post("/create_doc")
def create_doc(
    authorid: str = Form(None),
    document_title: str = Form(None),
    document_type: str = Form(None),
    document_content: str = Form(""),
    groups: str = Form(None), # TODO: Fix how these get submitted; current comma-separated list string
    is_published: bool = Form(False),
    current_user: models.User = Depends(get_current_user),
    analytics: Analytics = Depends(get_analytics),
):
    with compair.Session() as session:
        # Check if the trial has expired
        current_user = session.query(models.User).filter(models.User.user_id == current_user.user_id).first()
        trial_expiration = _trial_expiration(current_user)
        if HAS_TRIALS and trial_expiration and current_user.status == "trial" and trial_expiration < datetime.now(timezone.utc):
            current_user.status = "suspended"  # Mark as suspended once the trial expires
            current_user.status_change_date = datetime.now(timezone.utc)
            session.commit()

        # Enforce document limits (cloud plans) – core runs are unrestricted unless explicitly configured
        team = _user_team(current_user)
        document_limit: int | None = None
        if IS_CLOUD and HAS_TEAM and team and current_user.status == "active":
            document_limit = team.total_documents_limit  # type: ignore[union-attr]
        elif IS_CLOUD and _user_plan(current_user) == "individual" and current_user.status == "active":
            document_limit = 100
        else:
            raw_core_limit = os.getenv("COMPAIR_CORE_DOCUMENT_LIMIT")
            if raw_core_limit:
                try:
                    document_limit = int(raw_core_limit)
                except ValueError:
                    document_limit = None

        document_count = session.query(models.Document).filter(models.Document.user_id == current_user.user_id).count()

        if document_limit is not None and document_count >= document_limit:
            if IS_CLOUD:
                detail_msg = (
                    "Document limit reached. Individual plan users can have 100, team plans have 100 times "
                    "the number of users (pooled); other plans can have 10"
                )
            else:
                detail_msg = (
                    f"Document limit of {document_limit} reached. Adjust COMPAIR_CORE_DOCUMENT_LIMIT to raise "
                    "or unset it to remove limits in core deployments."
                )
            raise HTTPException(status_code=403, detail=detail_msg)

        if not authorid:
            authorid = current_user.user_id

        document = models.Document(
            user_id=current_user.user_id,
            author_id=authorid,
            title=document_title,
            content=document_content,
            doc_type=document_type,
            datetime_created=datetime.now(timezone.utc),
            datetime_modified=datetime.now(timezone.utc)
        )
        print('About to assign groups!')
        target_group_ids = []
        if groups:
            target_group_ids = [gid.strip() for gid in groups.split(',') if gid.strip()]

        if target_group_ids:
            q = select(models.Group).filter(models.Group.group_id.in_(target_group_ids))
            resolved_groups = session.execute(q).scalars().all()
            if not resolved_groups:
                raise HTTPException(status_code=404, detail="No matching groups found for provided IDs.")
            document.groups = resolved_groups
        else:
            q = select(models.Group).filter(models.Group.name == current_user.username)
            default_group = session.execute(q).scalars().first()
            if default_group is None:
                raise HTTPException(status_code=404, detail="Default group not found for user.")
            document.groups = [default_group]

        primary_group = document.groups[0]

        print(f'doc check!!! {document.content}')
        session.add(document)
        session.commit()

        if is_published:
            # Attempt to publish doc, pending user status check
            publish_doc(
                doc_id=document.document_id,
                is_published=True,
                current_user=current_user
            )

        # Log document creation
        log_activity(
            session=session, 
            user_id=document.author_id, 
            group_id=primary_group.group_id,
            action="create", 
            object_id=document.document_id, 
            object_name=document.title,
            object_type="document"
        )
        # Return the document_id for frontend use
        try:
            analytics.track("document_created", document.user_id)
        except Exception as exc:
            print(f"analytics track failed: {exc}")
        return {"document_id": document.document_id}


@router.get("/publish_doc")
def publish_doc(
    doc_id: str,
    is_published: bool,
    current_user: models.User = Depends(get_current_user)
):
    with compair.Session() as session:
        # Check if user can publish or not
        if current_user.status == "suspended":
            raise HTTPException(status_code=403, detail="Your subscription has expired. Renew to publish.")

        doc = session.query(models.Document).filter(
            models.Document.document_id == doc_id,
            models.Document.user_id == current_user.user_id
        )
        if not doc.first():
            raise HTTPException(status_code=404, detail="Document not found or you do not have permission to publish it.")
        if is_published!=doc.first().is_published:
            doc.update({'is_published': is_published})
            session.commit()


@router.get("/delete_docs")
def delete_docs(
    doc_ids: str,
    current_user: models.User = Depends(get_current_user)
):
    with compair.Session() as session:
        doc_ids = doc_ids.split(',')
        documents = session.query(models.Document).filter(
            models.Document.document_id.in_(doc_ids),
            models.Document.user_id == current_user.user_id
        )

        for document in documents:
            doc_id = document.document_id
            doc_name = document.title
            doc_group: models.Group = document.groups[0]
            group_id = doc_group.group_id
            log_activity(
                session=session, 
                user_id=current_user.user_id, 
                group_id=group_id,
                action="delete", 
                object_id=doc_id, 
                object_name=doc_name,
                object_type="document"
            )

        documents.delete()
        session.commit()


@router.get("/delete_doc")
def delete_doc(
    doc_id: str,
    current_user: models.User = Depends(get_current_user)
):
    with compair.Session() as session:
        document = session.query(models.Document).filter(
            models.Document.document_id == doc_id,
            models.Document.user_id == current_user.user_id
        )
        if not document.first():
            raise HTTPException(status_code=404, detail="Document not found or you do not have permission to delete it.")

        doc_group = document[0].groups[0]
        group_id = doc_group.group_id
        doc_name = document[0].title

        document.delete()
        session.commit()

        log_activity(
            session=session, 
            user_id=current_user.user_id, 
            group_id=group_id,
            action="delete", 
            object_id=doc_id, 
            object_name=doc_name,
            object_type="document"
        )


@router.post("/process_doc")
async def process_doc(
    doc_id: str = Form(...),
    doc_text: str = Form(...),
    generate_feedback: bool = Form(True),
    current_user: models.User = Depends(get_current_user),
    analytics: Analytics = Depends(get_analytics),
) -> Mapping[str, str | None]:
    with compair.Session() as session:
        doc = session.query(models.Document).filter(models.Document.document_id == doc_id).first()
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")
        # Only allow the author to process/edit
        if doc.author_id != current_user.user_id:
            raise HTTPException(status_code=403, detail="Only the author can edit this document")
        # If the user is suspended, allow user to edit, but not receive any new feedback for docs
        if current_user.status == "suspended":
            generate_feedback=False

    task_result = _dispatch_process_document_task(
        user_id=current_user.user_id,
        doc_id=doc_id,
        doc_text=doc_text,
        generate_feedback=generate_feedback,
    )
    task_id = getattr(task_result, "id", None)

    if generate_feedback:
        try:
            analytics.track("feedback_requested", current_user.user_id)
        except Exception as exc:
            print(f"analytics track failed: {exc}")
    return {"task_id": task_id}


@router.post("/upload/ocr-file")
async def upload_ocr_file(
    file: UploadFile = File(...),
    document_id: str = Form(None),
    current_user: models.User = Depends(get_current_user),
    settings: Settings = Depends(get_settings_dependency),
    ocr: OCRProvider = Depends(get_ocr),
):
    if not settings.ocr_enabled:
        raise HTTPException(status_code=501, detail="OCR is not available in this edition.")

    file_bytes = await file.read()
    try:
        task_id = ocr.submit(
            user_id=current_user.user_id,
            filename=file.filename or "upload",
            data=file_bytes,
            document_id=document_id,
        )
    except NotImplementedError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {"task_id": task_id}


@router.get("/ocr-file-result/{task_id}")
def get_ocr_file_result(
    task_id: str,
    current_user: models.User = Depends(get_current_user),
    settings: Settings = Depends(get_settings_dependency),
    ocr: OCRProvider = Depends(get_ocr),
):
    if not settings.ocr_enabled:
        raise HTTPException(status_code=501, detail="OCR is not available in this edition.")

    try:
        status = ocr.status(task_id)
    except NotImplementedError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if isinstance(status, dict):
        payload = {"task_id": task_id, **status}
    else:
        payload = {"task_id": task_id, "status": status}

    return payload


@router.get("/status/{task_id}")
async def get_process_status(
    task_id: str
):
    task_result = AsyncResult(task_id)
    print(task_result)
    print(task_result.status)
    if task_result.status == "SUCCESS":
        result = task_result.result
    elif task_result.status == "PENDING":
        result = "Task is still processing."
    else:
        result = "Task failed."
    return {"task_id": task_id, "status": task_result.status, "result": result}


@router.get("/trial_status")
def get_trial_status(
    current_user: models.User = Depends(get_current_user),
    billing: BillingProvider = Depends(get_billing),
    settings: Settings = Depends(get_settings_dependency),
):
    """Return the trial status for the authenticated user."""
    print('Trial 1')
    require_feature(HAS_TRIALS, "Trial management")
    with compair.Session() as session:
        trial_expiration = getattr(current_user, "trial_expiration_date", None)
        if trial_expiration is None:
            raise HTTPException(status_code=404, detail="Trial expiration date not found.")
        days_left = (trial_expiration - datetime.now(timezone.utc)).days
        show_banner = 0 < days_left <= 7

        checkout_url = None
        if show_banner and settings.billing_enabled:
            require_feature(HAS_BILLING, "Billing integration")
            db_user = session.query(models.User).filter(models.User.user_id == current_user.user_id).first()
            if db_user:
                if not db_user.stripe_customer_id:
                    try:
                        customer_id = billing.ensure_customer(
                            user_email=db_user.username,
                            user_id=db_user.user_id,
                        )
                    except NotImplementedError:
                        customer_id = None
                    else:
                        db_user.stripe_customer_id = customer_id
                        session.commit()
                customer_id = db_user.stripe_customer_id
                if customer_id:
                    try:
                        session_info = billing.create_checkout_session(
                            customer_id=customer_id,
                            price_id=plan_to_id.get("individual_monthly"),
                            qty=1,
                            success_url=settings.stripe_success_url,
                            cancel_url=settings.stripe_cancel_url,
                            metadata={"plan": "individual_monthly", "user_id": db_user.user_id},
                        )
                        checkout_url = getattr(session_info, "url", None)
                        if checkout_url is None and hasattr(session_info, "get"):
                            checkout_url = session_info.get("url")
                    except NotImplementedError:
                        checkout_url = None
                    except Exception as exc:
                        print(f"Error generating Checkout URL: {exc}")

        return {
            "status": current_user.status,
            "days_left": days_left,
            "show_banner": show_banner,
            "checkout_url": checkout_url,
        }


@router.get("/load_chunks")
def load_chunks(
    document_id: str,
) -> list[schema.Chunk]:
    with compair.Session() as session:
        chunks = session.query(models.Chunk).filter(
            models.Chunk.document_id==document_id
        )
        return chunks


@router.get("/load_feedback")
def load_feedback(
    chunk_id: str,
) -> schema.Feedback | None:
    with compair.Session() as session:
        feedback = session.query(models.Feedback).filter(
            models.Feedback.source_chunk_id==chunk_id
        )
        if len(feedback.all())>0:
            f = feedback[0]
            # Return user_feedback in the response
            return {
                "feedback_id": f.feedback_id,
                "source_chunk_id": f.source_chunk_id,
                "feedback": f.feedback,
                "user_feedback": f.user_feedback,
                "is_hidden": f.is_hidden,
            }
        return None


@router.post("/feedback/{feedback_id}/hide")
def hide_feedback(
    feedback_id: str,
    is_hidden: bool = Form(False)
):
    with compair.Session() as session:
        feedback = session.query(models.Feedback).filter(models.Feedback.feedback_id == feedback_id).first()
        if not feedback:
            raise HTTPException(status_code=404, detail="Feedback not found")
        feedback.is_hidden = is_hidden
        session.commit()
        return {"message": f"Feedback {'hidden' if is_hidden else 'unhidden'} successfully", "feedback_id": feedback_id}



@router.post("/feedback/{feedback_id}/rate")
def rate_feedback(
    feedback_id: str,
    user_feedback: str = Body(..., embed=True)  # expects "positive" or "negative"
):
    if user_feedback not in ("positive", "negative", None):
        raise HTTPException(status_code=400, detail="Invalid feedback label")
    with compair.Session() as session:
        feedback = session.query(models.Feedback).filter(models.Feedback.feedback_id == feedback_id).first()
        if not feedback:
            raise HTTPException(status_code=404, detail="Feedback not found")
        feedback.user_feedback = user_feedback
        session.commit()
        return {"message": "Feedback rated successfully", "feedback_id": feedback_id, "user_feedback": user_feedback}


@router.get("/documents/{document_id}/feedback")
def list_document_feedback(
    document_id: str,
    current_user: models.User = Depends(get_current_user)
):
    """Return feedback entries for a document the current user can access."""
    with compair.Session() as session:
        doc = session.query(models.Document).filter(models.Document.document_id == document_id).first()
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")
        # Access control: owner, member of a group with access, or published
        user_group_ids = {g.group_id for g in current_user.groups}
        doc_group_ids = {g.group_id for g in doc.groups}
        if not (doc.user_id == current_user.user_id or doc_group_ids & user_group_ids or doc.is_published):
            raise HTTPException(status_code=403, detail="Not authorized to access this document")
        # Join through chunks
        q = (
            session.query(models.Feedback)
            .join(models.Chunk, models.Feedback.source_chunk_id == models.Chunk.chunk_id)
            .filter(models.Chunk.document_id == document_id)
            .order_by(models.Feedback.timestamp.desc())
        )
        rows = q.all()
        return {
            "document_id": document_id,
            "count": len(rows),
            "feedback": [
                {
                    "feedback_id": f.feedback_id,
                    "chunk_id": f.source_chunk_id,
                    "feedback": f.feedback,
                    "user_feedback": f.user_feedback,
                    "timestamp": f.timestamp,
                } for f in rows
            ],
        }


@router.get("/load_references")
def load_references(
    chunk_id: str,
) -> list[schema.Reference]:
    with compair.Session() as session:
        references = session.query(models.Reference).filter(
            models.Reference.source_chunk_id==chunk_id
        ).all()
        returned_references: list[schema.Reference] = [
            schema.Reference(
                reference_id=r.reference_id,
                source_chunk_id=r.source_chunk_id,
                reference_document_id=r.reference_document_id,
                document=schema.Document(
                    document_id=r.document.document_id,
                    user_id=r.document.user_id,
                    author_id=r.document.author_id,
                    title=r.document.title,
                    content=r.document.content,
                    doc_type=r.document.doc_type,
                    datetime_created=r.document.datetime_created,
                    datetime_modified=r.document.datetime_modified,
                    is_published=r.document.is_published,
                    groups=[{"group_id":"","datetime_created":datetime.now(),"name":""}],
                    user=schema.User(
                        user_id=r.document.user.user_id,
                        username=r.document.user.username,
                        name=r.document.user.name,
                        groups=[{"group_id":"","datetime_created":datetime.now(),"name":""}],
                        datetime_registered=r.document.user.datetime_registered,
                        status=r.document.user.status
                    )
                ),
                document_author=r.document.user.username
            ) for r in references
        ]
        print(f'Returned references: {returned_references}')
        return returned_references

@router.get("/verify-email")
def verify_email(token: str):
    settings = get_settings_dependency()
    if not settings.require_authentication:
        raise HTTPException(status_code=403, detail="Email verification is disabled when authentication is disabled.")
    with compair.Session() as session:
        print(token)
        user = session.query(models.User).filter(models.User.verification_token == token).first()
        print(user)
        print(user.token_expiration)
        print(datetime.now(timezone.utc))
        if not user:
            raise HTTPException(status_code=400, detail="Invalid or expired token")
        if user.token_expiration < datetime.now(timezone.utc):
            raise HTTPException(status_code=400, detail="Token has expired")
        _activate_user_account(session, user, send_group_invites=True)
        return {"message": "Email verified successfully. Your free trial has started!"}

def is_valid_email(email):
    return re.match(r"[^@]+@[^@]+\.[^@]+", email)

@router.post("/sign-up")
def sign_up(
    request: schema.SignUpRequest,
    analytics: Analytics = Depends(get_analytics),
) -> dict:
    settings = get_settings_dependency()
    if not settings.require_authentication:
        raise HTTPException(status_code=403, detail="Sign-up is disabled when authentication is disabled.")
    print('1')
    if not is_valid_email(request.username):
        raise HTTPException(status_code=400, detail="Invalid email address")
    with compair.Session() as session:
        print('2')
        # Call internal function to create the user
        #user = create_user(email=request.email, password=request.password)
        user = create_user(
            username=request.username,
            name=request.name,
            password=request.password,
            groups=request.groups,
            session=session,
            referral_code=request.referral_code,
        )
        print('Passed create_user')
        if settings.require_email_verification:
            verification_link = f"http://{WEB_URL}/verify-email?token={user.verification_token}"
            print('3?')
            emailer.connect()
            print('4??')
            emailer.send(
                subject="Verify your email address",
                sender=EMAIL_USER,
                receivers=[user.username],
                html=_render_email(
                    ACCOUNT_VERIFY_TEMPLATE,
                    verification_link=verification_link,
                    user_name=user.name or user.username or "there",
                ),
            )
            print('The end???')
            return {"message": "Sign-up successful. Please check your email for verification."}
        _activate_user_account(session, user, send_group_invites=False)
        return {"message": "Sign-up successful. Your account is ready to use."}

@router.post("/forgot-password")
def forgot_password(request: schema.ForgotPasswordRequest) -> dict:
    settings = get_settings_dependency()
    if not settings.require_authentication:
        raise HTTPException(status_code=403, detail="Password resets are disabled when authentication is disabled.")
    print('1')
    with compair.Session() as session:
        print('2')
        user = session.query(models.User).filter(models.User.username == request.email).first()
        print(user)
        if not user:
            return {"message": "If the email exists, a reset link will be sent."}

        # Generate reset token
        token, expiration = generate_verification_token()  # Same function as before
        print(token)
        token = token.lower()
        user.reset_token = token
        user.token_expiration = expiration
        session.commit()
        
        print('3')
        # Send email with reset link
        reset_link = f"http://{WEB_URL}/reset-password?token={token}"
        emailer.connect()
        print('4')
        emailer.send(
            subject="Password Reset Request",
            sender=EMAIL_USER,
            receivers=[request.email],
            html=_render_email(
                PASSWORD_RESET_TEMPLATE,
                reset_link=reset_link,
                user_name=user.name or user.username or "",
            ),
        )
        print('5')
        return {"message": "If the email exists, a reset link will be sent."}

@router.post("/reset-password")
def reset_password(request: schema.ResetPasswordRequest) -> dict:
    settings = get_settings_dependency()
    if not settings.require_authentication:
        raise HTTPException(status_code=403, detail="Password resets are disabled when authentication is disabled.")
    with compair.Session() as session:
        print('1')
        print(request.token)
        user = session.query(models.User).filter(models.User.reset_token == request.token).first()
        print(user)
        if not user or user.token_expiration < datetime.now(timezone.utc):
            raise HTTPException(status_code=400, detail="Invalid or expired token")
        
        # Update the password
        user.set_password(request.new_password)
        print('2')
        user.reset_token = None  # Invalidate the token
        user.token_expiration = None
        print('3')
        session.commit()
        print('4')
        
        return {"message": "Password has been reset successfully"}

@router.get("/admin/groups")
def get_admin_groups(
    current_user: models.User = Depends(get_current_user),
):
    """Retrieve groups managed by the given user (admin)."""
    with compair.Session() as session:
        admin = session.query(models.Administrator).filter(models.Administrator.user_id == current_user.user_id).first()
        if not admin:
            return []  # Not an admin of any groups

        # Only return groups where this user is an admin
        groups = admin.groups
        return [
            {
                "group_id": g.group_id,
                "name": g.name,
                "visibility": g.visibility,
                "category": g.category,
                "description": g.description,
                "group_image": g.group_image,
                "datetime_created": g.datetime_created,
            }
            for g in groups
        ]

@router.get("/admin/join_requests")
def get_join_requests(
    group_id: str,
    current_user: models.User = Depends(get_current_user)
):
    """Retrieve pending join requests for a group."""
    with compair.Session() as session:
        requests = session.query(models.JoinRequest).filter(
            models.JoinRequest.group_id == group_id
        ).filter(
            models.JoinRequest.group.has(models.Group.admins.any(models.Administrator.user_id == current_user.user_id))
        ).all()
        return [
            {"request_id": r.request_id, "user_name": r.user.name, "datetime_requested": r.datetime_requested}
            for r in requests
        ]

@router.post("/admin/approve_request")
def approve_request(
        request_id: int,
        current_user: models.User = Depends(get_current_user)
    ):
    """Approve a join request."""
    with compair.Session() as session:
        request = session.query(models.JoinRequest).filter(
            models.JoinRequest.request_id == request_id
        ).filter(
            models.JoinRequest.group.has(models.Group.admins.any(models.Administrator.user_id == current_user.user_id))
        ).first()
        if not request:
            raise HTTPException(status_code=404, detail="Request not found")

        group: models.Group = request.group
        user: models.User = request.user
        group.users.append(user)
        session.delete(request)
        session.commit()

        log_activity(
            session=session, 
            user_id=user.user_id, 
            group_id=group.group_id,
            action="join", 
            object_id=group.group_id, 
            object_name=group.name,
            object_type="group"
        )

        return {"message": "Request approved successfully"}

@router.post("/admin/reject_request")
def reject_request(
    request_id: int,
    current_user: models.User = Depends(get_current_user)
):
    """Reject a join request."""
    with compair.Session() as session:
        request = session.query(models.JoinRequest).filter(
            models.JoinRequest.request_id == request_id,
            models.JoinRequest.group.has(models.Group.admins.any(models.Administrator.user_id == current_user.user_id))
        ).first()
        if not request:
            raise HTTPException(status_code=404, detail="Request not found")

        session.delete(request)
        session.commit()

        return {"message": "Request rejected successfully"}

@router.post("/admin/update_group")
def update_group(
    group_id: str, 
    name: Optional[str] = Form(None), 
    visibility: Optional[str] = Form(None), 
    category: Optional[str] = Form(None), 
    description: Optional[str] = Form(None),
    current_user: models.User = Depends(get_current_user)
) -> dict:
    """Update group settings."""
    print('1')
    with compair.Session() as session:
        group = session.query(models.Group).filter(
            models.Group.group_id == group_id,
            models.Group.admins.any(models.Administrator.user_id == current_user.user_id)
        ).first()
        print('2')
        if not group:
            raise HTTPException(status_code=404, detail="Group not found")
        print(group)
        print(group_id)
        print(name)
        print(visibility)
        print(category)
        print(description)
        if name:
            group.name = name
        if visibility:
            group.visibility = visibility
        if category:
            group.category = category
        if description:
            group.description = description
        session.commit()

        return {"message": "Group updated successfully"}

@router.post("/admin/invite_member")
def invite_member_to_group(
    request: schema.InviteMemberRequest,
    current_user: models.User = Depends(get_current_user)
):
    """
    Invite an existing Compair user (by username or email) to join a group.
    If the user exists, create a GroupInvitation and send an email.
    """
    admin_id = request.admin_id, 
    group_id = request.group_id, 
    username = request.username
    with compair.Session() as session:
        group = session.query(models.Group).filter(
            models.Group.group_id == group_id,
            models.Group.admins.any(models.Administrator.user_id == admin_id)
        ).first()
        if not group:
            raise HTTPException(status_code=404, detail="Group not found")

        admin = session.query(models.Administrator).filter(
            models.Administrator.user_id == admin_id,
            models.Administrator.groups.contains(group)
        ).first()
        if not admin:
            raise HTTPException(status_code=403, detail="You are not authorized to manage this group")

        # Try to find user by username or email
        user = session.query(models.User).filter(
            (models.User.username == username)# | (models.User.email == username_or_email) # if they were different
        ).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        # Check if already a member or already invited
        if user in group.users:
            raise HTTPException(status_code=400, detail="User is already a member")
        existing_invite = session.query(models.GroupInvitation).filter(
            models.GroupInvitation.group_id == group_id,
            models.GroupInvitation.email == user.username
        ).first()
        if existing_invite:
            raise HTTPException(status_code=400, detail="User already invited")

        # Create invitation
        token = secrets.token_urlsafe(32).lower()
        invitation = models.GroupInvitation(
            group_id=group_id,
            inviter_id=admin_id,
            token=token,
            email=user.username,
            datetime_expiration=datetime.utcnow() + timedelta(days=7),
            status='pending'
        )
        session.add(invitation)
        session.commit()

        # Send email
        invitation_link = f"http://{WEB_URL}/accept-group-invitation?token={token}&user_id={user.user_id}"
        emailer.connect()
        emailer.send(
            subject="You’re Invited to Join a Group on Compair",
            sender=EMAIL_USER,
            receivers=[user.username],
            html=GROUP_INVITATION_TEMPLATE.replace(
                "{{ inviter_name }}", admin.user.name
            ).replace(
                "{{ group_name }}", group.name
            ).replace(
                "{{ invitation_link }}", invitation_link
            )
        )
        invitation.status = 'sent'
        session.commit()
        return {"message": "Invitation sent successfully"}

@router.post("/admin/invite_new_user")
def invite_new_user_to_group(
    request: schema.InviteToGroupRequest,
    current_user: models.User = Depends(get_current_user)
) -> dict:
    admin_id = request.admin_id
    group_id = request.group_id
    email = request.email
    """Generate an invitation link to Compair and send it via email, logging a Group request to send on signup."""
    with compair.Session() as session:
        group = session.query(models.Group).filter(
            models.Group.group_id == group_id,
            models.Group.admins.any(models.Administrator.user_id == admin_id)
        ).first()
        if not group:
            raise HTTPException(status_code=404, detail="Group not found")

        admin = session.query(models.Administrator).filter(
            models.Administrator.user_id == admin_id,
            models.Administrator.groups.contains(group)
        ).first()
        if not admin:
            raise HTTPException(status_code=403, detail="You are not authorized to manage this group")

        referral_link = generate_referral_link(admin.user.referral_code)

        # Send email notification
        emailer.connect()
        # Track pending group invitation for this email if group_id is provided
        if group_id:
            # Store a "pending" group invitation for this email
            token = secrets.token_urlsafe(32).lower()
            invitation = models.GroupInvitation(
                group_id=group_id,
                inviter_id=admin.user.user_id,
                token=token,
                email=email,
                datetime_expiration=datetime.utcnow() + timedelta(days=7),
                status="pending"
            )
            session.add(invitation)
            session.commit()
        # Send email notification
        emailer.send(
            subject="You're Invited to Compair!",
            sender=EMAIL_USER,
            receivers=[email],
            html=INDIVIDUAL_INVITATION_TEMPLATE.replace(
                "{{inviter_name}}", admin.user.name
            ).replace(
                "{{referral_link}}", referral_link,
            )
        )

        return {"message": "Invitation to join Compair successful."}

@router.post("/admin/remove_member")
def remove_member(
    request: schema.RemoveMemberRequest,
    current_user: models.User = Depends(get_current_user)
):
    """Remove a member from a group."""
    with compair.Session() as session:
        # Validate that the group exists
        group = session.query(models.Group).filter(models.Group.group_id == request.group_id).first()
        if not group:
            raise HTTPException(status_code=404, detail="Group not found")

        # Validate that the current user is an admin of the group
        if not any(admin.user_id == current_user.user_id for admin in group.admins):
            raise HTTPException(status_code=403, detail="You are not authorized to manage this group")

        # Validate that the user to be removed is a member of the group
        user = session.query(models.User).filter(models.User.user_id == request.user_id).first()
        if not user or user not in group.users:
            raise HTTPException(status_code=404, detail="User is not a member of this group")

        # Remove the user from the group
        group.users.remove(user)
        session.commit()

        # Log the activity
        log_activity(
            session=session,
            user_id=current_user.user_id,
            group_id=group.group_id,
            action="remove_member",
            object_id=user.user_id,
            object_name=user.name,
            object_type="user"
        )

        return {"message": f"User {user.name} has been removed from the group"}

@router.get("/accept_group_invitation")
def accept_group_invitation(
    token: str, 
    current_user: models.User = Depends(get_current_user)
):
    """Accept a group invitation using a token."""
    with compair.Session() as session:
        invitation = session.query(models.GroupInvitation).filter(models.GroupInvitation.token == token).first()
        if not invitation:
            raise HTTPException(status_code=404, detail="Invalid or expired invitation")

        if invitation.datetime_expiration < datetime.now(timezone.utc):
            invitation.status = "expired"
            session.commit()
            raise HTTPException(status_code=400, detail="Invitation has expired")

        group = invitation.group
        user = session.query(models.User).filter(models.User.user_id == current_user.user_id).first()

        # Add the user to the group
        group.users.append(user)
        invitation.status = "accepted"
        session.commit()

        return {"message": "You have successfully joined the group"}

@router.get("/accept_team_invitation")
def accept_team_invitation(
    token: str, 
    current_user: models.User = Depends(get_current_user)
):
    """Accept a team invitation using a token."""
    require_feature(HAS_TEAM, "Team collaboration")
    with compair.Session() as session:
        invitation = session.query(models.TeamInvitation).filter(
            models.TeamInvitation.invitation_id == token,
            models.TeamInvitation.status == "sent",
        ).first()
        if not invitation:
            raise HTTPException(status_code=404, detail="Invalid or expired invitation.")

        if hasattr(current_user, "team_id"):
            current_user.team_id = invitation.team_id
        invitation.status = "accepted"
        session.commit()

        return {"message": "You have successfully joined the team!"}

def get_feedback_tooltip(
    feedback_id: str,
) -> str:
    """Retrieve feedback details for tooltip display."""
    with compair.Session() as session:
        feedbacks = session.query(models.Feedback).filter(models.Feedback.feedback_id == feedback_id).all()
        return " | ".join(f.feedback for f in feedbacks)

@router.get("/get_activity_feed")
def get_activity_feed(
    user_id: str, 
    page: int = 1, 
    page_size: int = 10, 
    include_own_activities: bool = True,
    current_user: models.User = Depends(get_current_user)
):
    """Retrieve recent activities for a user's groups."""
    require_feature(HAS_ACTIVITY, "Activity feed")
    if not IS_CLOUD:
        raise HTTPException(status_code=501, detail="Activity feed is only available in the Compair Cloud edition.")
    with compair.Session() as session:
        # Get user's groups
        
        # Query recent activities related to user's groups
        group_ids = [g.group_id for g in current_user.groups]

        q = (
            session.query(models.Activity)
            .filter(models.Activity.group_id.in_(group_ids))
            .order_by(models.Activity.timestamp.desc())
        )
        if not include_own_activities:
            q = q.filter(models.Activity.user_id != current_user.user_id)

        total_count = q.count()
        offset = (page - 1) * page_size
        activities = q.offset(offset).limit(page_size).all()

        return {
            "activities": [
                {
                    "user": activity.user.name,
                    "user_id": activity.user_id,
                    "group_id": activity.group_id,
                    "action": activity.action,
                    "object": f"{activity.object_type} {activity.object_name}",
                    "object_type": activity.object_type,
                    "object_name": activity.object_name,
                    "object_id": activity.object_id,
                    "timestamp": activity.timestamp,
                    "tooltip": get_feedback_tooltip(activity.object_id) if activity.action == "provided feedback" else None
                }
                for activity in activities
            ],
            "total_count": total_count
        }

@router.delete("/delete_group")
def delete_group(
    group_id: str,
    current_user: models.User = Depends(get_current_user)
):
    """Delete a group. Allowed only if the current user is an admin of the group.

    This removes the group and its associations. Documents remain but lose their link to this group.
    """
    with compair.Session() as session:
        group = session.query(models.Group).filter(models.Group.group_id == group_id).first()
        if not group:
            raise HTTPException(status_code=404, detail="Group not found")
        # Check admin rights
        is_admin = any(a.user_id == current_user.user_id for a in group.admins)
        if not is_admin:
            raise HTTPException(status_code=403, detail="Not authorized to delete this group")
        name = group.name
        session.delete(group)
        session.commit()
        # Log activity
        try:
            log_activity(
                session=session,
                user_id=current_user.user_id,
                group_id=group_id,
                action="delete",
                object_id=group_id,
                object_name=name,
                object_type="group",
            )
        except Exception:
            pass
        return {"message": "Group deleted", "group_id": group_id}

plan_to_id = {
    #"starter": "price_1Rpb5VEPPPghXLIJkgJyWBXb",
    #"standard": "price_1Qv7etEPPPghXLIJ5SR8KgXD",
    "individual_monthly": "price_1Ry34UEPPPghXLIJ4OLQNEna",#"price_1RwnRaEPPPghXLIJIPyeVxLq",
    "individual_annual": "price_1Ry34NEPPPghXLIJeftgEYOD",#"price_1RwoQkEPPPghXLIJyNUCRRo3",
    "team_monthly": "price_1Ry34HEPPPghXLIJoEyGRm7h",#"price_1RwoRDEPPPghXLIJt7uTCC7E",
    "team_annual": "price_1Ry34REPPPghXLIJXhLhwpBW"#"price_1RwoP9EPPPghXLIJo16OPgMu",
}

@router.post("/create-checkout-session")
def create_checkout_session(
    checkout_map: Mapping,
    current_user: models.User = Depends(get_current_user),
    billing: BillingProvider = Depends(get_billing),
    settings: Settings = Depends(get_settings_dependency),
):
    """Create a checkout session through the configured billing provider."""

    require_cloud("Billing")

    if not settings.billing_enabled:
        raise HTTPException(status_code=501, detail="Billing is not available in this edition.")

    plan = checkout_map.get("plan", "individual_monthly")
    team_emails = checkout_map.get("team_emails", []) or []
    if isinstance(team_emails, str):
        team_emails = [email.strip() for email in team_emails.split(',') if email.strip()]
    if plan not in plan_to_id:
        raise HTTPException(status_code=400, detail="Unsupported plan selected.")

    if plan.startswith("team") and len(team_emails) < 3:
        raise HTTPException(status_code=400, detail="Team plans require at least 4 emails.")

    price_id = plan_to_id[plan]

    quantity = 1 if plan.startswith("individual") else len(team_emails)

    try:
        with compair.Session() as session:
            db_user = session.query(models.User).filter(models.User.user_id == current_user.user_id).first()
            if not db_user:
                raise HTTPException(status_code=404, detail="User not found.")

            if not db_user.stripe_customer_id:
                log_event("stripe_customer_create_attempt", user_id=db_user.user_id)
                try:
                    customer_id = billing.ensure_customer(
                        user_email=db_user.username,
                        user_id=db_user.user_id,
                    )
                except NotImplementedError as exc:
                    raise HTTPException(status_code=501, detail=str(exc)) from exc
                db_user.stripe_customer_id = customer_id
                session.commit()
                log_event("stripe_customer_created", user_id=db_user.user_id, customer_id=customer_id)

            customer_id = db_user.stripe_customer_id

            if plan.startswith("team"):
                require_feature(HAS_TEAM, "Team collaboration")
                team_name = checkout_map.get("team_name", f"{db_user.name}'s Team")
                team = models.Team(
                    name=team_name,
                    total_documents_limit=100 * len(team_emails),
                    daily_feedback_limit=50 * len(team_emails),
                )
                session.add(team)
                session.commit()

                for email in team_emails:
                    existing_user = session.query(models.User).filter(models.User.username == email).first()
                    if not existing_user:
                        invitation = models.TeamInvitation(
                            email=email,
                            inviter_id=db_user.user_id,
                            team_id=team.team_id,
                            status="pending",
                            datetime_created=datetime.now(timezone.utc),
                        )
                        session.add(invitation)
                session.commit()

            log_event("stripe_checkout_session_start", user_id=db_user.user_id)

            try:
                session_info = billing.create_checkout_session(
                    customer_id=customer_id,
                    price_id=price_id,
                    qty=quantity,
                    success_url=settings.stripe_success_url,
                    cancel_url=settings.stripe_cancel_url,
                    metadata={"plan": plan, "user_id": db_user.user_id},
                )
            except NotImplementedError as exc:
                raise HTTPException(status_code=501, detail=str(exc)) from exc
            except Exception as exc:
                raise HTTPException(status_code=500, detail=str(exc)) from exc

            session_id = getattr(session_info, "id", None)
            session_url = getattr(session_info, "url", None)
            if session_id is None and hasattr(session_info, "get"):
                session_id = session_info.get("id")
            if session_url is None and hasattr(session_info, "get"):
                session_url = session_info.get("url")
            if not session_id or not session_url:
                raise HTTPException(status_code=500, detail="Billing provider did not return a checkout URL.")

            if hasattr(db_user, "checkout_session"):
                db_user.checkout_session = session_id
            if hasattr(db_user, "plan"):
                db_user.plan = plan
            session.commit()
            log_event("stripe_checkout_session_created", user_id=db_user.user_id, session_id=session_id)
            return {"url": session_url}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/redirect-to-checkout")
def redirect_to_checkout(
    user_id: str,
    billing: BillingProvider = Depends(get_billing),
    settings: Settings = Depends(get_settings_dependency),
):
    require_cloud("Billing")
    if not settings.billing_enabled:
        raise HTTPException(status_code=404, detail="Not found")

    with compair.Session() as session:
        user = session.query(models.User).filter(models.User.user_id == user_id).first()
        if not user or not user.checkout_session:
            raise HTTPException(status_code=500, detail=f"No checkout session found for user {user_id}")

        session_id = user.checkout_session
        try:
            checkout_url = billing.get_checkout_url(session_id)
        except NotImplementedError as exc:
            raise HTTPException(status_code=501, detail=str(exc)) from exc

        user.checkout_session = None
        session.commit()

    return {"url": checkout_url}



def track_payment_method(fingerprint: str):
    """
    Track the number of credits associated with a payment fingerprint.
    """
    with compair.Session() as session:
        total_credits = session.query(models.User).filter(
            models.User.payment_fingerprint == fingerprint,
            models.User.referral_credits > 0
        ).count()

        max_credits_per_payment = 3
        if total_credits >= max_credits_per_payment:
            raise HTTPException(
                status_code=400,
                detail="Referral credit limit reached for this payment method."
            )

def handle_successful_checkout(session):
    """
    Capture the payment method during Stripe Checkout.
    """
    print('made it to checkout event')
    print(session)
    customer_id = session["customer"]
    if customer_id is None:
        customer_id = session["customer_details"]["email"]

    payment_method = session.get("payment_method")  # Might not exist if using trial
    print(customer_id)
    print(payment_method)
    if payment_method:
        with compair.Session() as session:
            user = session.query(models.User).filter(
                models.User.stripe_customer_id == customer_id
            ).first()
            print(user)
            if user:
                user.payment_fingerprint = payment_method
                track_payment_method(user.payment_fingerprint)
                session.commit()
    
    payment_status = session["payment_status"]
    if payment_status != "paid":
        return
    
    with compair.Session() as db_session:
        user = db_session.query(models.User).filter(
            models.User.stripe_customer_id == customer_id
        ).first()
        if not user:
            return

        if user.plan.startswith("team"):
            team = db_session.query(models.Team).filter(
                models.Team.ownership.has(owner_id=user.user_id)
            ).first()
            if not team:
                return

            # Update invitations with the new team ID and send emails
            invitations = db_session.query(models.TeamInvitation).filter(
                models.TeamInvitation.inviter_id == user.user_id,
                models.TeamInvitation.status == "pending",
            ).all()
            for invitation in invitations:
                existing_user = db_session.query(models.User).filter(
                    models.User.username == invitation.email
                ).first()

                if existing_user:
                    # Send team invitation email to existing user
                    emailer.connect()
                    emailer.send(
                        subject="You're Invited to Join a Team on Compair",
                        sender=EMAIL_USER,
                        receivers=[invitation.email],
                        html=f"""
                        <p>{user.name} has invited you to join their team on Compair!</p>
                        <p>Click <a href="https://{WEB_URL}/accept-invitation?token={invitation.invitation_id}">here</a> to join.</p>
                        """
                    )
                    invitation.status = "sent"
                else:
                    # Send invitation to join Compair
                    send_invite(
                        invitee_emails=invitation.email,
                        current_user=user,
                    )

def handle_successful_payment_intent(intent):
    """
    Capture the payment method when the first invoice is paid.
    """
    ### Get / Store payment intent to customer_id (if both available!) if payment_method not available
    customer_id = intent["customer"]
    payment_method = intent.get("payment_method")
    print("Got to intent")
    print(intent)
    print(payment_method)
    if payment_method:
        with compair.Session() as session:
            user = session.query(models.User).filter(
                models.User.stripe_customer_id == customer_id
            ).first()
            if user and hasattr(user, "payment_fingerprint"):
                user.payment_fingerprint = payment_method
                track_payment_method(user.payment_fingerprint)
                session.commit()

def handle_successful_invoice_payment(
    invoice,
    billing: BillingProvider,
    analytics: Analytics | None = None,
):
    """
    Capture the payment method when the first invoice is paid.
    """
    ### Use payment_intent to lookup customer (if latter is not available) to store payment_method
    customer_id = invoice["customer"]
    price_id = None
    for line in invoice["lines"]["data"]:
        if "price" in line and "id" in line["price"]:
            price_id = line["price"]["id"]
            break
    
    plan = None
    for k, v in plan_to_id.items():
        if v == price_id:
            plan = k
            break

    payment_method = invoice.get("payment_method")
    print("Got to payment")
    print(invoice)
    print(payment_method)
    if payment_method:
        with compair.Session() as session:
            user = session.query(models.User).filter(
                models.User.stripe_customer_id == customer_id
            ).first()
            if user:
                user.payment_fingerprint = payment_method
                track_payment_method(user.payment_fingerprint)
                session.commit()

    with compair.Session() as session:
        user = session.query(models.User).filter(
            models.User.stripe_customer_id == customer_id
        ).first()
        print(f'invoice user: {user}')
        if user:
            # user.stripe_subscription_id = subscription_id ## Can track if needed
            user.status = 'active'  # Mark the user as subscribed
            if hasattr(user, "last_payment_date"):
                user.last_payment_date = datetime.now(timezone.utc)
            user.status_change_date = datetime.now(timezone.utc)
            if plan and hasattr(user, "plan"):
                user.plan = plan
            print(user.status)
            # Handle referrer logic
            if hasattr(user, "referred_by") and user.referred_by:
                print(user.referred_by)
                referrer = session.query(models.User).filter(
                    models.User.user_id == user.referred_by
                ).first()
                print(referrer)
                if referrer and hasattr(referrer, "pending_referral_credits") and referrer.pending_referral_credits > 0:
                    # Convert pending credits to earned credits
                    if hasattr(referrer, "referral_credits"):
                        referrer.referral_credits += 10
                    referrer.pending_referral_credits -= 10
                    # Send email notification
                    emailer.connect()
                    emailer.send(
                        subject="You've Earned a Free Month!",
                        sender=EMAIL_USER,
                        receivers=[referrer.username],
                        html=REFERRAL_CREDIT_TEMPLATE.replace(
                            "{{user_name}}", referrer.name
                        ).replace(
                            "{{referral_credits}}", str(referrer.referral_credits),
                        )
                    )
                    # Create and apply a $15 coupon in Stripe ($25 if team plan)
                    amount = 15
                    if referrer.plan == 'team':
                        amount=25
                    try:
                        coupon_id = billing.create_coupon(amount)
                        print(f'Coupon ID 2:{coupon_id}')
                        if referrer.stripe_customer_id:
                            billing.apply_coupon(customer_id=referrer.stripe_customer_id, coupon_id=coupon_id)
                    except NotImplementedError:
                        pass

            if analytics:
                try:
                    analytics.track("subscription_created", user.user_id)
                except Exception as exc:
                    print(f"analytics track failed: {exc}")

            session.commit()

def handle_subscription_cancellation(subscription):
    """
    Revoke access if a subscription is canceled.
    """
    customer_id = subscription["customer"]
    print('Cancelled')
    print(subscription)
    with compair.Session() as session:
        user = session.query(models.User).filter(
            models.User.stripe_customer_id == customer_id
        ).first()

        if user:
            user.status = 'suspended'  # Remove access
            user.status_change_date = datetime.now(timezone.utc)
            session.commit()

def handle_failed_payment(invoice):
    """
    Handle failed payment attempts.
    """
    customer_id = invoice["customer"]
    print('Failure')
    print(invoice)
    with compair.Session() as session:
        user = session.query(models.User).filter(
            models.User.stripe_customer_id == customer_id
        ).first()

        if user:
            user.status = 'suspended'  # Suspend access until payment is made
            user.status_change_date = datetime.now(timezone.utc)
            session.commit()

@router.post("/webhook")
async def stripe_webhook(
    request: Request,
    billing: BillingProvider = Depends(get_billing),
    analytics: Analytics = Depends(get_analytics),
    settings: Settings = Depends(get_settings_dependency),
):
    if not settings.billing_enabled:
        raise HTTPException(status_code=404, detail="Not found")

    payload = await request.body()
    sig_header = request.headers.get("stripe-signature")
    #print(f'payload: {payload}')
    try:
        event = billing.construct_event(payload, sig_header)
        print(f'webhook event: {event["type"]}')

        log_event("stripe_webhook_start", event_type=event['type'], data=event['data']['object'])

        if event["type"] == "payment_intent.succeeded":
            intent = event["data"]["object"]
            handle_successful_payment_intent(intent)

        elif event["type"] == "invoice.payment_succeeded":
            invoice = event["data"]["object"]
            handle_successful_invoice_payment(invoice, billing, analytics)

        elif event["type"] == "customer.subscription.deleted":
            subscription = event["data"]["object"]
            handle_subscription_cancellation(subscription)

        elif event["type"] == "invoice.payment_failed":
            invoice = event["data"]["object"]
            handle_failed_payment(invoice)

        log_event("stripe_webhook_complete", event_type=event['type'], data=event['data']['object'])

        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))



def generate_referral_link(referral_code: str) -> str:
    """
    Generate a referral link using the user's referral code.
    """
    require_cloud("Referral program")
    base_url = f"http://{WEB_URL}/login"
    return f"{base_url}?ref={referral_code}"

@router.post("/send-invite")
def send_invite(
    invitee_emails: str = Form(...),
    group_id: str = Form(None),
    current_user: models.User = Depends(get_current_user),
):
    """
    Send an email invitation with the user's referral code.
    If group_id is provided, track the invitation for group auto-invite on sign-up.
    """
    require_cloud("Referral program")
    invitee_emails = invitee_emails.split(',')
    with compair.Session() as session:

        referral_link = generate_referral_link(current_user.referral_code)

        # Send email notification
        emailer.connect()
        for invitee_email in invitee_emails:
            # Track pending group invitation for this email if group_id is provided
            if group_id:
                # Store a "pending" group invitation for this email
                token = secrets.token_urlsafe(32).lower()
                invitation = models.GroupInvitation(
                    group_id=group_id,
                    inviter_id=current_user.user_id,
                    token=token,
                    email=invitee_email,
                    datetime_expiration=datetime.utcnow() + timedelta(days=7),
                    status="pending"
                )
                session.add(invitation)
                session.commit()
            # Send email notification
            emailer.send(
                subject="You're Invited to Compair!",
                sender=EMAIL_USER,
                receivers=[invitee_email],
                html=INDIVIDUAL_INVITATION_TEMPLATE.replace(
                    "{{inviter_name}}", current_user.name
                ).replace(
                    "{{referral_link}}", referral_link,
                )
            )

        return {"message": "Invitation sent successfully"}

@router.post("/get-customer-portal")
def get_customer_portal(
    current_user: models.User = Depends(get_current_user),
    billing: BillingProvider = Depends(get_billing),
    settings: Settings = Depends(get_settings_dependency),
):
    """Generate a billing portal session link through the billing provider."""

    if not settings.billing_enabled:
        raise HTTPException(status_code=501, detail="Billing is not available in this edition.")

    return_url = f"http://{WEB_URL}/home" if WEB_URL else "https://compair.sh/home"

    try:
        with compair.Session() as session:
            user = session.query(models.User).filter(models.User.user_id == current_user.user_id).first()
            if not user or not user.stripe_customer_id:
                raise HTTPException(status_code=400, detail="No billing profile found for this user.")

            try:
                portal_url = billing.create_customer_portal(
                    customer_id=user.stripe_customer_id,
                    return_url=return_url,
                )
            except NotImplementedError as exc:
                raise HTTPException(status_code=501, detail=str(exc)) from exc

            return {"url": portal_url}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc




@router.post("/upload/profile")
async def upload_profile_image(
    upload_type: str = Form(...),
    file: UploadFile = File(...),
    current_user: models.User = Depends(get_current_user)
) -> Mapping[str, str]:
    if upload_type!='profile':
        raise HTTPException(status_code=400, detail="Invalid upload type")

    allowed_types = ["image/jpg", "image/jpeg", "image/png", "image/webp"]
    if file.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail="Invalid file type")

    # Upload to Cloudflare Images
    headers = {
        "Authorization": f"Bearer {CLOUDFLARE_IMAGES_TOKEN}"
    }
    data = {
        "requireSignedURLs": "false"
    }
    files = {
        "file": (file.filename, await file.read(), file.content_type)
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(
            CLOUDFLARE_IMAGES_UPLOAD_URL,
            headers=headers,
            data=data,
            files=files,
            timeout=30
        )
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail="Cloudflare Images upload failed")

    result = response.json()
    if not result.get("success"):
        raise HTTPException(status_code=500, detail="Cloudflare Images upload failed")

    image_id = result["result"]["id"]

    with compair.Session() as session:
        user = session.query(models.User).filter(models.User.user_id == current_user.user_id).first()
        user.profile_image = image_id
        session.commit()
    
    # Read file bytes to get size
    file_bytes = await file.read()
    file_size = len(file_bytes)

    # Log the upload event
    log_event(
        "cloudflare_upload",
        upload_type="profile_image",
        user_id=current_user.user_id,
        file_size=file_size,
        file_key=image_id,
        content_type=file.content_type
    )

    return {"image_id": image_id}


@router.get("/get_profile_image")
def get_profile_image(
    variant: str = Query("public", enum=["public", "avatar", "preview"]),
    current_user: models.User = Depends(get_current_user)
) -> Mapping[str, str]:
    with compair.Session() as session:
        image_id = current_user.profile_image
        if not image_id:
            raise HTTPException(status_code=400, detail="No profile image found")
        image_url = f"{CLOUDFLARE_IMAGES_BASE_URL}/{image_id}/{variant}"
    return {"url": image_url}


@router.post("/upload/file")
async def upload_file(
    document_id: str = Form(...),
    upload_type: str = Form(...),
    file: UploadFile = File(...),
    current_user: models.User = Depends(get_current_user),
    storage: StorageProvider = Depends(get_storage),
) -> Mapping[str, str]:
    if upload_type != 'file':
        raise HTTPException(status_code=400, detail="Invalid upload type")

    allowed_types = ["application/pdf"]#, "document/docx"]
    if file.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail="Invalid file type")

    file_name = file.filename
    file_name_hash = hashlib.sha256(f"{file_name}".encode()).hexdigest()
    file_key = f"{upload_type}/{current_user.user_id}/{file_name_hash}/{document_id}"

    # Read file bytes to get size and reset file pointer before upload
    file_bytes = await file.read()
    file_size = len(file_bytes)
    await file.seek(0)

    content_type = file.content_type or "application/octet-stream"

    try:
        upload_url = storage.put_file(file_key, file.file, content_type)
    except NotImplementedError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - surfaced to client
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    with compair.Session() as session:
        # Fetch the document
        document = session.query(models.Document).filter(models.Document.document_id == document_id).first()
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")

        document.file_key = file_key
        session.commit()
    
    # Log the upload event
    log_event(
        "cloudflare_upload",
        upload_type="document_file",
        user_id=current_user.user_id,
        document_id=document_id,
        file_size=file_size,
        file_key=file_key,
        content_type=content_type
    )

    return {"url": upload_url}


@router.post("/upload/group")
async def upload_group_image(
    group_id: str,
    upload_type: str,
    file: UploadFile = File(...)
) -> Mapping[str, str]:
    if upload_type!='group':
        raise HTTPException(status_code=400, detail="Invalid upload type")

    allowed_types = ["image/jpg", "image/jpeg", "image/png", "image/webp"]
    if file.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail="Invalid file type")

    # Upload to Cloudflare Images
    headers = {
        "Authorization": f"Bearer {CLOUDFLARE_IMAGES_TOKEN}"
    }
    data = {
        "requireSignedURLs": "false"
    }
    files = {
        "file": (file.filename, await file.read(), file.content_type)
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(
            CLOUDFLARE_IMAGES_UPLOAD_URL,
            headers=headers,
            data=data,
            files=files,
            timeout=30
        )
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail="Cloudflare Images upload failed")

    result = response.json()
    if not result.get("success"):
        raise HTTPException(status_code=500, detail="Cloudflare Images upload failed")

    image_id = result["result"]["id"]

    with compair.Session() as session:
        group = session.query(models.Group).filter(models.Group.group_id == group_id).first()
        group.group_image = image_id
        session.commit()

    # Read file bytes to get size
    file_bytes = await file.read()
    file_size = len(file_bytes)

    # Log the upload event
    log_event(
        "cloudflare_upload",
        upload_type="group_image",
        group_id=group_id,
        file_size=file_size,
        file_key=image_id,
        content_type=file.content_type
    )

@router.post("/send-feature-announcement")
def send_feature_announcement(admin_key: str = Header(None)):
    """
    Manually trigger a feature announcement email to churned users.
    Requires an admin API key.
    """
    if not IS_CLOUD or send_feature_announcement_task is None:
        raise HTTPException(status_code=501, detail="Feature announcements require the Compair Cloud edition.")
    if admin_key != ADMIN_API_KEY:
        raise HTTPException(status_code=403, detail="Unauthorized")

    send_feature_announcement_task.delay()  # Run email sending as an async Celery task

    return {"message": "Feature announcement email campaign triggered successfully."}


@router.post("/retrial/{user_id}")
def grant_retrial(user_id: str):
    """
    Reactivate a churned user's trial for 2 weeks.
    """
    if not IS_CLOUD:
        raise HTTPException(status_code=501, detail="Re-trials require the Compair Cloud edition.")
    with Session() as session:
        user = session.query(models.User).filter(models.User.user_id == user_id).first()

        if not user or user.status != "suspended":
            raise HTTPException(status_code=400, detail="User not eligible for a re-trial.")

        if user.retrial_count >= 1:
            raise HTTPException(status_code=403, detail="Re-trial limit reached.")

        # Reactivate trial
        user.status = "trial"
        user.status_change_date = datetime.now(timezone.utc)
        user.trial_expiration_date = datetime.now(timezone.utc) + timedelta(days=14)
        user.retrial_count += 1  # Increment re-trial count
        user.last_retrial_date = datetime.now(timezone.utc)
        session.commit()

        return {"message": "Your re-trial has been activated!"}

@router.post("/documents/{document_id}/notes")
def create_note(
    document_id: str,
    content: str = Form(...),
    group_id: str = Form(None),
    current_user: models.User = Depends(get_current_user)
):
    """Create a Note for a Document. User must be in the Document's group. Also chunk/embed Note content."""
    with compair.Session() as session:
        document = session.query(models.Document).filter(models.Document.document_id == document_id).first()
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        # Check group membership
        doc_group_ids = [g.group_id for g in document.groups]
        user_group_ids = [g.group_id for g in current_user.groups]
        if not set(doc_group_ids) & set(user_group_ids):
            raise HTTPException(status_code=403, detail="User not in document's group")
        note = models.Note(
            document_id=document_id,
            author_id=current_user.user_id,
            group_id=group_id or (doc_group_ids[0] if doc_group_ids else None),
            content=content,
            datetime_created=datetime.now(timezone.utc)
        )
        session.add(note)
        session.commit()
        session.refresh(note)
        # Chunk and embed Note content
        note_text_chunks = chunk_text(content)
        embedder = Embedder()
        for text in note_text_chunks:
            chunk_hash = hash(text)
            embedding = create_embedding(embedder, text, user=current_user)
            note_chunk = models.Chunk(
                hash=str(chunk_hash),
                document_id=document_id,
                note_id=note.note_id,
                chunk_type="note",
                content=text,
            )
            note_chunk.embedding = embedding
            session.add(note_chunk)
        session.commit()

        log_activity(
            session=session,
            user_id=current_user.user_id,
            group_id=note.group_id,
            action="create",
            object_id=note.note_id,
            object_name=f"Note on {document.title}",
            object_type="note"
        )

        return schema.Note(
            note_id=note.note_id,
            document_id=note.document_id,
            author_id=note.author_id,
            group_id=note.group_id,
            content=note.content,
            datetime_created=note.datetime_created,
            author=schema.User(
                user_id=current_user.user_id,
                username=current_user.username,
                name=current_user.name,
                datetime_registered=current_user.datetime_registered,
                status=current_user.status,
                profile_image=current_user.profile_image,
                role=current_user.role,
            )
        )

@router.get("/documents/{document_id}/notes")
def list_notes(
    document_id: str,
    current_user: models.User = Depends(get_current_user)
) -> list[schema.Note]:
    """List all Notes for a Document."""
    with compair.Session() as session:
        notes = session.query(models.Note).filter(models.Note.document_id == document_id).all()
        result = []
        for note in notes:
            author = session.query(models.User).filter(models.User.user_id == note.author_id).first()
            result.append(schema.Note(
                note_id=note.note_id,
                document_id=note.document_id,
                author_id=note.author_id,
                group_id=note.group_id,
                content=note.content,
                datetime_created=note.datetime_created,
                author=schema.User(
                    user_id=author.user_id,
                    username=author.username,
                    name=author.name,
                    datetime_registered=author.datetime_registered,
                    status=author.status,
                    profile_image=author.profile_image,
                    role=author.role,
                ) if author else None
            ))
        return result

@router.get("/notes/{note_id}")
def get_note(
    note_id: str,
    current_user: models.User = Depends(get_current_user)
) -> schema.Note:
    """Get a single Note by ID."""
    with compair.Session() as session:
        note = session.query(models.Note).filter(models.Note.note_id == note_id).first()
        if not note:
            raise HTTPException(status_code=404, detail="Note not found")
        author = session.query(models.User).filter(models.User.user_id == note.author_id).first()
        return schema.Note(
            note_id=note.note_id,
            document_id=note.document_id,
            author_id=note.author_id,
            group_id=note.group_id,
            content=note.content,
            datetime_created=note.datetime_created,
            author=schema.User(
                user_id=author.user_id,
                username=author.username,
                name=author.name,
                datetime_registered=author.datetime_registered,
                status=author.status,
                profile_image=author.profile_image,
                role=author.role,
            ) if author else None
        )

@router.get("/documents/{document_id}/file-url")
def get_document_file_url(
    document_id: str,
    current_user: models.User = Depends(get_current_user),
    storage: StorageProvider = Depends(get_storage),
):
    """Return the file URL for a Document."""
    with compair.Session() as session:
        doc = session.query(models.Document).filter(models.Document.document_id == document_id).first()
        if not doc or not doc.file_key:
            raise HTTPException(status_code=404, detail="File not found for this document.")
        file_url = storage.build_url(doc.file_key)
        return {"file_url": file_url}

@router.get("/documents/{document_id}/image-url")
def get_document_image_url(
    document_id: str,
    current_user: models.User = Depends(get_current_user),
    storage: StorageProvider = Depends(get_storage),
):
    """Return the preview image URL for a Document."""
    with compair.Session() as session:
        doc = session.query(models.Document).filter(models.Document.document_id == document_id).first()
        if not doc or not doc.image_key:
            raise HTTPException(status_code=404, detail="Image not found for this document.")
        image_url = storage.build_url(doc.image_key)
        return {"image_url": image_url}


def sanitize_filename(filename: str) -> str:
    # Replace non-ASCII characters with underscore
    return re.sub(r'[^\x00-\x7F]+', '_', filename)


@router.post("/documents/{document_id}/generate-download-token")
def generate_download_token(
    document_id: str,
    current_user: models.User = Depends(get_current_user)
):
    # Check permissions as in your download endpoint
    with compair.Session() as session:
        doc = session.query(models.Document).filter(models.Document.document_id == document_id).first()
        if not doc or not doc.file_key:
            raise HTTPException(status_code=404, detail="File not found for this document.")

        # Authorization check (same as download endpoint)
        if current_user.user_id == doc.author_id:
            pass
        elif doc.is_published:
            doc_group_ids = {g.group_id for g in doc.groups}
            user_group_ids = {g.group_id for g in current_user.groups}
            if not doc_group_ids & user_group_ids:
                raise HTTPException(status_code=403, detail="Not authorized to download this file.")
        else:
            raise HTTPException(status_code=403, detail="Not authorized to download this file.")

    if not HAS_REDIS:
        raise HTTPException(status_code=501, detail="Secure download links require Redis, which is unavailable in the core edition.")

    token = secrets.token_urlsafe(32)
    key = f"download_token:{token}"
    redis_client.setex(key, 300, document_id)
    return {"download_url": f"/documents/download/{token}"}


@router.get("/documents/download/{token}")
def download_document_with_token(
    token: str,
    storage: StorageProvider = Depends(get_storage),
):
    if not HAS_REDIS:
        raise HTTPException(status_code=501, detail="Secure download links require Redis, which is unavailable in the core edition.")

    key = f"download_token:{token}"
    value = redis_client.get(key) if redis_client else None
    document_id = value.decode('utf-8') if value else None
    if not document_id:
        raise HTTPException(status_code=403, detail="Invalid or expired token")
    redis_client.delete(key)
    with compair.Session() as session:
        doc = session.query(models.Document).filter(models.Document.document_id == document_id).first()
        if not doc or not doc.file_key:
            raise HTTPException(status_code=404, detail="File not found for this document.")

        safe_title = sanitize_filename(doc.title or 'file')
        try:
            file_obj, content_type = storage.get_file(doc.file_key)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Stored file could not be located.") from exc

        return StreamingResponse(
            file_obj,
            media_type=content_type,
            headers={
                "Content-Disposition": f"attachment; filename={safe_title or 'file'}"
            }
        )


@router.get("/documents/{document_id}/download")
def download_document_file(
    document_id: str,
    current_user: models.User = Depends(get_current_user),
    storage: StorageProvider = Depends(get_storage),
):
    with compair.Session() as session:
        doc = session.query(models.Document).filter(models.Document.document_id == document_id).first()

        safe_title = sanitize_filename(doc.title or 'file')

        if not doc or not doc.file_key:
            raise HTTPException(status_code=404, detail="File not found for this document.")

        # Authorization check
        if current_user.user_id == doc.author_id:
            pass
        elif doc.is_published:
            # Check group membership
            doc_group_ids = {g.group_id for g in doc.groups}
            user_group_ids = {g.group_id for g in current_user.groups}
            if not doc_group_ids & user_group_ids:
                raise HTTPException(status_code=403, detail="Not authorized to download this file.")

        # Fetch file from R2 and stream to user
        try:
            file_obj, content_type = storage.get_file(doc.file_key)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Stored file could not be located.") from exc

        return StreamingResponse(
            file_obj,
            media_type=content_type,
            headers={
                "Content-Disposition": f"attachment; filename={safe_title or 'file'}"
            }
        )


@router.post("/help-request")
def submit_help_request(
    content: str = Form(...),
    current_user: models.User = Depends(get_current_user)
):
    with compair.Session() as session:
        help_request = models.HelpRequest(
            user_id=current_user.user_id,
            content=content,
            datetime_created=datetime.now(timezone.utc)
        )
        session.add(help_request)
        session.commit()
        if not IS_CLOUD:
            raise HTTPException(status_code=501, detail="Help request emails require the Compair Cloud edition.")
        send_help_request_email.delay(help_request.request_id)
        return {"message": "Your request has been submitted. Our team will get back to you soon."}


@router.post("/deactivate-account")
def submit_deactivate_request(
    notice: str = Form(...),
    current_user: models.User = Depends(get_current_user)
):
    with compair.Session() as session:
        deactivate_request = models.DeactivateRequest(
            user_id=current_user.user_id,
            notice=notice,
            datetime_created=datetime.now(timezone.utc)
        )
        session.add(deactivate_request)
        session.commit()
        if not IS_CLOUD:
            raise HTTPException(status_code=501, detail="Deactivate request emails require the Compair Cloud edition.")
        send_deactivate_request_email.delay(deactivate_request.request_id)
        return {"message": f"We’ve received your request and will delete your account and data shortly. If you change your mind, reach out within 24 hours at {EMAIL_USER}."}


CORE_PATHS: set[str] = {
    "/sign-up",
    "/verify-email",
    "/login",
    "/forgot-password",
    "/reset-password",
    "/load_session",
    "/update_user",
    "/load_groups",
    "/load_group",
    "/create_group",
    "/join_group",
    "/load_group_users",
    "/delete_group",
    "/load_documents",
    "/load_document",
    "/load_document_by_id",
    "/load_user_files",
    "/create_doc",
    "/update_doc",
    "/publish_doc",
    "/delete_doc",
    "/delete_docs",
    "/process_doc",
    "/status/{task_id}",
    "/upload/ocr-file",
    "/ocr-file-result/{task_id}",
    "/load_chunks",
    "/load_references",
    "/load_feedback",
    "/documents/{document_id}/feedback",
    "/get_activity_feed",
}

for route in router.routes:
    if isinstance(route, APIRoute) and route.path in CORE_PATHS:
        core_router.routes.append(route)


def create_fastapi_app():
    """Backwards-compatible app factory for running this module directly."""
    from fastapi import FastAPI

    fastapi_app = FastAPI()
    fastapi_app.include_router(router)
    return fastapi_app


app = create_fastapi_app()
