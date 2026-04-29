# database.py

import psycopg2
import os
from contextlib import contextmanager


# =============================================================================
# Connection
# =============================================================================

def get_connection():
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        database=os.getenv("DB_NAME", "chatbot_db"),
        user=os.getenv("DB_USER", "chatbot"),
        password=os.getenv("DB_PASSWORD", "chatbot123")
    )


@contextmanager
def get_cursor():
    conn = get_connection()
    cur = conn.cursor()
    try:
        yield cur
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        cur.close()
        conn.close()


# =============================================================================
# USERS
# =============================================================================

def create_user(first_name, last_name=None, age=None):
    with get_cursor() as cur:
        cur.execute(
            """
            INSERT INTO users (first_name, last_name, age)
            VALUES (%s, %s, %s)
            RETURNING id
            """,
            (first_name, last_name, age)
        )
        return cur.fetchone()[0]


def get_user(user_id):
    with get_cursor() as cur:
        cur.execute("SELECT * FROM users WHERE id=%s", (user_id,))
        return cur.fetchone()


# =============================================================================
# CONVERSATIONS
# =============================================================================

def create_conversation(user_id):
    with get_cursor() as cur:
        cur.execute(
            """
            INSERT INTO conversations (user_id)
            VALUES (%s)
            RETURNING id
            """,
            (user_id,)
        )
        return cur.fetchone()[0]


def get_last_conversation(user_id):
    with get_cursor() as cur:
        cur.execute(
            """
            SELECT id FROM conversations
            WHERE user_id=%s
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (user_id,)
        )
        result = cur.fetchone()
        return result[0] if result else None


def update_conversation_topic(conversation_id, topic):
    with get_cursor() as cur:
        cur.execute(
            """
            UPDATE conversations
            SET last_topic=%s
            WHERE id=%s
            """,
            (topic, conversation_id)
        )


def update_conversation_summary(conversation_id, summary):
    with get_cursor() as cur:
        cur.execute(
            """
            UPDATE conversations
            SET summary=%s
            WHERE id=%s
            """,
            (summary, conversation_id)
        )


def get_last_topic(conversation_id):
    with get_cursor() as cur:
        cur.execute(
            "SELECT last_topic FROM conversations WHERE id=%s",
            (conversation_id,)
        )
        result = cur.fetchone()
        return result[0] if result else "unknown"


def get_summary(conversation_id):
    with get_cursor() as cur:
        cur.execute(
            "SELECT summary FROM conversations WHERE id=%s",
            (conversation_id,)
        )
        result = cur.fetchone()
        return result[0] if result else ""


# =============================================================================
# MESSAGES
# =============================================================================

def save_message(conversation_id, role, content):
    with get_cursor() as cur:
        cur.execute(
            """
            INSERT INTO messages (conversation_id, role, content)
            VALUES (%s, %s, %s)
            """,
            (conversation_id, role, content)
        )


def get_last_messages(conversation_id, limit=5):
    with get_cursor() as cur:
        cur.execute(
            """
            SELECT role, content
            FROM messages
            WHERE conversation_id=%s
            ORDER BY created_at DESC
            LIMIT %s
            """,
            (conversation_id, limit)
        )
        rows = cur.fetchall()

        # tagasta õiges järjekorras
        return list(reversed(rows))


# =============================================================================
# USER MEMORY
# =============================================================================

def set_user_memory(user_id, key, value):
    with get_cursor() as cur:
        cur.execute(
            """
            INSERT INTO user_memory (user_id, key, value)
            VALUES (%s, %s, %s)
            ON CONFLICT (user_id, key)
            DO UPDATE SET value = EXCLUDED.value
            """,
            (user_id, key, value)
        )


def get_user_memory(user_id, key):
    with get_cursor() as cur:
        cur.execute(
            """
            SELECT value FROM user_memory
            WHERE user_id=%s AND key=%s
            """,
            (user_id, key)
        )
        result = cur.fetchone()
        return result[0] if result else None


def get_all_user_memory(user_id):
    with get_cursor() as cur:
        cur.execute(
            """
            SELECT key, value FROM user_memory
            WHERE user_id=%s
            """,
            (user_id,)
        )
        return dict(cur.fetchall())


# =============================================================================
# HELPER: init flow
# =============================================================================

def get_or_create_conversation(user_id):
    conversation_id = get_last_conversation(user_id)
    if conversation_id:
        return conversation_id
    return create_conversation(user_id)
