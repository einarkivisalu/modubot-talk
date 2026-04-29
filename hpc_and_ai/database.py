import psycopg2
import os

def get_connection():
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        database=os.getenv("DB_NAME", "chatbot_db"),
        user=os.getenv("DB_USER", "chatbot"),
        password=os.getenv("DB_PASSWORD", "chatbot123")
    )

def create_user(name):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        "INSERT INTO users (first_name) VALUES (%s) RETURNING id",
        (name,)
    )

    user_id = cur.fetchone()[0]
    conn.commit()
    cur.close()
    conn.close()

    return user_id

def create_conversation(user_id):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        "INSERT INTO conversations (user_id) VALUES (%s) RETURNING id",
        (user_id,)
    )

    conv_id = cur.fetchone()[0]
    conn.commit()
    cur.close()
    conn.close()

    return conv_id

def save_message(conversation_id, role, text):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        "INSERT INTO messages (conversation_id, role, content) VALUES (%s, %s, %s)",
        (conversation_id, role, text)
    )

    conn.commit()
    cur.close()
    conn.close()

def get_last_topic(conversation_id):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        "SELECT last_topic FROM conversations WHERE id=%s",
        (conversation_id,)
    )

    result = cur.fetchone()

    cur.close()
    conn.close()

    return result[0] if result else "unknown"

def update_topic(conversation_id, topic):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        "UPDATE conversations SET last_topic=%s WHERE id=%s",
        (topic, conversation_id)
    )

    conn.commit()
    cur.close()
    conn.close()
