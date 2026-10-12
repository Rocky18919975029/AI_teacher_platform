"""SQLite WAL metadata, isolated sessions and durable ingestion queue.

One application process per database is the supported deployment topology.
Short transactions also make job transitions atomic across threads.
"""

import json
import sqlite3
import time
import uuid
from contextlib import contextmanager


class Store:
    def __init__(self, path):
        self.path = path
        with self.db() as db:
            db.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY, owner TEXT NOT NULL, course TEXT NOT NULL,
                    created REAL NOT NULL, busy INTEGER NOT NULL DEFAULT 0);
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY, session TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                    role TEXT NOT NULL, content TEXT NOT NULL, created REAL NOT NULL,
                    metadata TEXT NOT NULL DEFAULT '{}');
                CREATE INDEX IF NOT EXISTS messages_session ON messages(session, id);
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY, owner TEXT NOT NULL, course TEXT NOT NULL,
                    filename TEXT NOT NULL, path TEXT NOT NULL, status TEXT NOT NULL,
                    attempts INTEGER NOT NULL DEFAULT 0, result TEXT, error TEXT,
                    created REAL NOT NULL, updated REAL NOT NULL);
                CREATE INDEX IF NOT EXISTS jobs_pending ON jobs(status, created);
                CREATE TABLE IF NOT EXISTS audio (
                    id TEXT PRIMARY KEY, owner TEXT NOT NULL, course TEXT NOT NULL, created REAL NOT NULL);
                PRAGMA user_version=1;
            """)

    @contextmanager
    def db(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        db.execute('PRAGMA foreign_keys=ON')
        try:
            with db:
                yield db
        finally:
            db.close()

    def session(self, sid, owner):
        with self.db() as db:
            row = db.execute('SELECT * FROM sessions WHERE id=? AND owner=?', (sid, owner)).fetchone()
            if row is None:
                raise KeyError('Session not found')
            result = dict(row)
            result['messages'] = [
                dict(r)
                for r in db.execute(
                    'SELECT role,content,metadata FROM (SELECT id,role,content,metadata FROM messages WHERE session=? ORDER BY id DESC LIMIT 100) ORDER BY id',
                    (sid,),
                )
            ]
            for message in result['messages']:
                message.update(json.loads(message.pop('metadata')))
            first = db.execute(
                "SELECT content FROM messages WHERE session=? AND role='user' ORDER BY id LIMIT 1", (sid,)
            ).fetchone()
            result['title'] = first[0][:70] if first else 'New conversation'
            return result

    def create_session(self, owner, course):
        sid = str(uuid.uuid4())
        with self.db() as db:
            db.execute(
                'INSERT INTO sessions(id,owner,course,created) VALUES(?,?,?,?)',
                (sid, owner, course, time.time()),
            )
        return sid

    def claim_session(self, sid, owner):
        with self.db() as db:
            return (
                db.execute(
                    'UPDATE sessions SET busy=1 WHERE id=? AND owner=? AND busy=0',
                    (sid, owner),
                ).rowcount
                == 1
            )

    def release_session(self, sid):
        with self.db() as db:
            db.execute('UPDATE sessions SET busy=0 WHERE id=?', (sid,))

    def append_turn(self, sid, prompt, answer, metadata=None):
        with self.db() as db:
            db.executemany(
                'INSERT INTO messages(session,role,content,created,metadata) VALUES(?,?,?,?,?)',
                [
                    (sid, 'user', prompt, time.time(), '{}'),
                    (sid, 'assistant', answer, time.time(), json.dumps(metadata or {})),
                ],
            )

    def sessions(self, owner, course):
        with self.db() as db:
            return [
                dict(r)
                for r in db.execute(
                    "SELECT id,course,created, (SELECT substr(content,1,70) FROM messages WHERE session=sessions.id AND role='user' ORDER BY messages.id LIMIT 1) AS title FROM sessions WHERE owner=? AND course=? ORDER BY created DESC LIMIT 100",
                    (owner, course),
                )
            ]

    def delete_session(self, sid, owner):
        with self.db() as db:
            return (
                db.execute('DELETE FROM sessions WHERE id=? AND owner=? AND busy=0', (sid, owner)).rowcount
                == 1
            )

    def enqueue(self, owner, course, filename, path, limit):
        jid = str(uuid.uuid4())
        with self.db() as db:
            db.execute('BEGIN IMMEDIATE')
            pending = db.execute(
                "SELECT COUNT(*) FROM jobs WHERE status IN ('pending','running')"
            ).fetchone()[0]
            if pending >= limit:
                raise OverflowError('Ingestion queue is full')
            db.execute(
                'INSERT INTO jobs(id,owner,course,filename,path,status,created,updated) VALUES(?,?,?,?,?,?,?,?)',
                (jid, owner, course, filename, str(path), 'pending', time.time(), time.time()),
            )
        return self.job(jid)

    def job(self, jid):
        with self.db() as db:
            row = db.execute('SELECT * FROM jobs WHERE id=?', (jid,)).fetchone()
            if row is None:
                raise KeyError('Job not found')
            result = dict(row)
            if result['result']:
                result['result'] = json.loads(result['result'])
            return result

    def recover_jobs(self):
        with self.db() as db:
            db.execute('UPDATE sessions SET busy=0')
            db.execute("UPDATE jobs SET status='pending' WHERE status='running' AND attempts<3")
            db.execute(
                "UPDATE jobs SET status='failed',error='Retry limit reached after restart' WHERE status='running'"
            )

    def claim_job(self):
        with self.db() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute(
                "SELECT * FROM jobs WHERE status='pending' AND updated<=? ORDER BY created LIMIT 1",
                (time.time(),),
            ).fetchone()
            if not row:
                return None
            db.execute(
                "UPDATE jobs SET status='running',attempts=attempts+1,updated=? WHERE id=?",
                (time.time(), row['id']),
            )
            return dict(row)

    def active_uploads(self):
        with self.db() as db:
            return {
                row[0] for row in db.execute("SELECT path FROM jobs WHERE status IN ('pending','running')")
            }

    def finish_job(self, jid, result=None, error=None, retry=False):
        with self.db() as db:
            attempts = db.execute('SELECT attempts FROM jobs WHERE id=?', (jid,)).fetchone()[0]
            retry = bool(error and retry and attempts < 3)
            db.execute(
                'UPDATE jobs SET status=?,result=?,error=?,updated=? WHERE id=?',
                (
                    'pending' if retry else ('failed' if error else 'completed'),
                    json.dumps(result) if result else None,
                    error,
                    time.time() + (2**attempts if retry else 0),
                    jid,
                ),
            )

    def register_audio(self, aid, owner, course):
        with self.db() as db:
            db.execute('INSERT INTO audio VALUES(?,?,?,?)', (aid, owner, course, time.time()))

    def get_audio(self, aid, owner, ttl):
        with self.db() as db:
            row = db.execute(
                'SELECT * FROM audio WHERE id=? AND owner=? AND created>?', (aid, owner, time.time() - ttl)
            ).fetchone()
            return dict(row) if row else None

    def expire_audio(self, cutoff):
        with self.db() as db:
            ids = [r[0] for r in db.execute('SELECT id FROM audio WHERE created<?', (cutoff,))]
            db.execute('DELETE FROM audio WHERE created<?', (cutoff,))
            return ids
