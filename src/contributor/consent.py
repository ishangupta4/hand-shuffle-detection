"""Consent record creation."""

import datetime
import uuid


def create_consent_id():
    return str(uuid.uuid4())[:8]


def make_consent_record(session_id, consent_id, user_agent, collection_mode, storage_backend):
    return {
        "consent_id": consent_id,
        "session_id": session_id,
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "user_agent": user_agent,
        "collection_mode": collection_mode,
        "storage_backend": storage_backend,
    }
