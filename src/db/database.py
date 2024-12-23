import sqlite3
from typing import List, Dict, Any


class Database:
    def __init__(self):
        self.conn = sqlite3.connect("srma.db", check_same_thread=False)
        self.cursor = self.conn.cursor()
        self._create_tables()

    def _create_tables(self):
        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS abstract_answers (
                ensemble_id TEXT,
                abstract_id TEXT,
                abstract_answer TEXT,
                PRIMARY KEY (ensemble_id, abstract_id)
            )
        """
        )

        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS batch_ids (
                ensemble_id TEXT PRIMARY KEY,
                batch_ids TEXT,
                model TEXT
            )
        """
        )
        self.conn.commit()

    def store_abstract_answers(
        self, ensemble_id: str, abstract_id: str, abstract_answer: str
    ):
        self.cursor.execute(
            "INSERT OR REPLACE INTO abstract_answers VALUES (?, ?, ?)",
            (ensemble_id, abstract_id, abstract_answer),
        )
        self.conn.commit()

    def store_multiple_abstract_answers(
        self, ensemble_id: str, abstract_ids: List[str], abstract_answers: List[str]
    ):
        self.cursor.executemany(
            "INSERT OR REPLACE INTO abstract_answers VALUES (?, ?, ?)",
            [
                (ensemble_id, aid, answer)
                for aid, answer in zip(abstract_ids, abstract_answers)
            ],
        )
        self.conn.commit()

    def retrieve_abstract_answers(
        self, ensemble_id: str, abstract_id: int
    ) -> List[Dict[str, Any]]:
        self.cursor.execute(
            "SELECT * FROM abstract_answers WHERE ensemble_id = ? AND abstract_id = ?",
            (ensemble_id, abstract_id),
        )
        rows = self.cursor.fetchall()
        return [
            {"ensemble_id": row[0], "abstract_id": row[1], "abstract_answer": row[2]}
            for row in rows
        ]

    def retrieve_multiple_abstract_answers(
        self, ensemble_id: str, abstract_ids: List[str]
    ) -> Dict[str, str]:
        self.cursor.execute(
            "SELECT * FROM abstract_answers WHERE ensemble_id = ? AND abstract_id IN ({})".format(
                ",".join(map(str, abstract_ids))
            ),
            (ensemble_id,),
        )
        rows = self.cursor.fetchall()
        return {row[1]: row[2] for row in rows}

    def store_batch_ids(self, ensemble_id: str, batch_ids: List[str], model: str):
        batch_ids_str = ",".join(batch_ids)
        self.cursor.execute(
            "INSERT OR REPLACE INTO batch_ids VALUES (?, ?, ?)",
            (ensemble_id, batch_ids_str, model),
        )
        self.conn.commit()

    def retrieve_batch_ids(self, ensemble_id: str) -> Dict[str, Any]:
        self.cursor.execute(
            "SELECT * FROM batch_ids WHERE ensemble_id = ?", (ensemble_id,)
        )
        row = self.cursor.fetchone()
        return {"batch_ids": row[1].split(","), "model": row[2]}
