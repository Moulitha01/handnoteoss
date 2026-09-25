"""SQLite storage with FTS5 full-text search (so old notes become searchable)."""
import os
import sqlite3
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "notes.db")


def conn():
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    return c


def init():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    with conn() as c:
        c.execute("""CREATE TABLE IF NOT EXISTS notes(
            id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT, text TEXT,
            tags TEXT DEFAULT '', engine TEXT, image TEXT, created TEXT)""")
        # migration: add new columns to an existing notes.db
        existing = {r["name"] for r in c.execute("PRAGMA table_info(notes)")}
        if "summary" not in existing:
            c.execute("ALTER TABLE notes ADD COLUMN summary TEXT DEFAULT ''")
        if "original_text" not in existing:
            c.execute("ALTER TABLE notes ADD COLUMN original_text TEXT DEFAULT ''")
        c.execute("CREATE VIRTUAL TABLE IF NOT EXISTS notes_fts USING fts5(title, text, tags)")


def _sync_fts(c, nid, title, text, tags):
    c.execute("DELETE FROM notes_fts WHERE rowid=?", (nid,))
    c.execute("INSERT INTO notes_fts(rowid,title,text,tags) VALUES(?,?,?,?)", (nid, title, text, tags))


def add(title, text, engine, image, tags="", summary="", original_text=""):
    with conn() as c:
        cur = c.execute(
            "INSERT INTO notes(title,text,tags,engine,image,created,summary,original_text) "
            "VALUES(?,?,?,?,?,?,?,?)",
            (title, text, tags, engine, image, datetime.now().strftime("%Y-%m-%d %H:%M"),
             summary, original_text))
        _sync_fts(c, cur.lastrowid, title, text, tags)
        return cur.lastrowid


def update(nid, title, text, tags, summary=None):
    with conn() as c:
        c.execute("UPDATE notes SET title=?,text=?,tags=? WHERE id=?", (title, text, tags, nid))
        if summary is not None:
            c.execute("UPDATE notes SET summary=? WHERE id=?", (summary, nid))
        _sync_fts(c, nid, title, text, tags)


def delete(nid):
    with conn() as c:
        c.execute("DELETE FROM notes WHERE id=?", (nid,))
        c.execute("DELETE FROM notes_fts WHERE rowid=?", (nid,))


def get(nid):
    with conn() as c:
        r = c.execute("SELECT * FROM notes WHERE id=?", (nid,)).fetchone()
        return dict(r) if r else None


def search(q=""):
    with conn() as c:
        if q.strip():
            term = " ".join(w + "*" for w in q.replace('"', " ").split())
            rows = c.execute("""SELECT n.* FROM notes n JOIN notes_fts f ON f.rowid=n.id
                                WHERE notes_fts MATCH ? ORDER BY rank""", (term,)).fetchall()
        else:
            rows = c.execute("SELECT * FROM notes ORDER BY id DESC").fetchall()
        return [dict(r) for r in rows]