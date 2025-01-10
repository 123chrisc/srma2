import sqlite3
from typing import List, Dict, Any, Optional
import json
from src.api.data_extraction import VariableDefinition

class Database:
    def __init__(self, path="data_extraction.db"):
        # Use Row objects to allow row["column_name"] access
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.cursor = self.conn.cursor()
        self._create_tables()

    def _create_tables(self):
        # Store batch processing information
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS batch_info (
                ensemble_id TEXT PRIMARY KEY,
                batch_ids TEXT,
                model TEXT,
                retrieval_status TEXT DEFAULT 'pending',
                variables TEXT,
                doc_ids TEXT,
                dataset_path TEXT,
                preprompt TEXT,
                prompt TEXT
            )
        """)

        # Store extraction results
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS extraction_results (
                ensemble_id TEXT,
                article_id TEXT,
                extraction_text TEXT,
                PRIMARY KEY (ensemble_id, article_id)
            )
        """)
        
        # Store variable extractions
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS extracted_variables (
                ensemble_id TEXT,
                article_id TEXT,
                variable_name TEXT,
                extracted_value TEXT,
                PRIMARY KEY (ensemble_id, article_id, variable_name)
            )
        """)
        
        self.conn.commit()

    def store_extraction_result(self, ensemble_id: str, article_id: str, extraction_text: str):
        self.cursor.execute(
            "INSERT OR REPLACE INTO extraction_results VALUES (?, ?, ?)",
            (ensemble_id, article_id, extraction_text)
        )
        self.conn.commit()

    def store_multiple_extraction_results(self, ensemble_id: str, article_id: List[str], extraction_texts: List[str]):
        self.cursor.executemany(
            "INSERT OR REPLACE INTO extraction_results VALUES (?, ?, ?)",
            [(ensemble_id, doc_id, text) for doc_id, text in zip(article_id, extraction_texts)]
        )
        self.conn.commit()

    def store_batch_info(
        self,
        ensemble_id: str,
        batch_ids: List[str],
        model: str,
        variables: List[VariableDefinition],
        doc_ids: List[str],
        dataset_path: str,
        preprompt: str,
        prompt: str
    ) -> None:
        # Convert variables to JSON
        variables_json = json.dumps([v.dict() for v in variables])

        # Join batch_ids and doc_ids into comma-separated for storage
        batch_ids_str = ",".join(batch_ids)
        doc_ids_str = ",".join(doc_ids) if doc_ids else ""

        self.cursor.execute("""
            INSERT INTO batch_info (
                ensemble_id,
                batch_ids,
                model,
                retrieval_status,
                variables,
                doc_ids,
                dataset_path,
                preprompt,
                prompt
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            ensemble_id,
            batch_ids_str,
            model,
            "pending",             # or whatever your default retrieval status is
            variables_json,
            doc_ids_str,
            dataset_path,          # store dataset_path here
            preprompt,
            prompt
        ))
        self.conn.commit()

    def store_variable_extraction(self, ensemble_id: str, article_id: str, variable_name: str, extracted_value: str):
        self.cursor.execute(
            "INSERT OR REPLACE INTO extracted_variables VALUES (?, ?, ?, ?)",
            (ensemble_id, article_id, variable_name, extracted_value)
        )
        self.conn.commit()

    def store_multiple_variable_extractions(self, ensemble_id: str, article_id: str, extractions: Dict[str, Any]):
        # Stores each variable separately
        for var_name, value in extractions.items():
            self.cursor.execute(
                "INSERT OR REPLACE INTO extracted_variables VALUES (?, ?, ?, ?)",
                (ensemble_id, article_id, var_name, str(value))
            )
        self.conn.commit()

    def retrieve_extraction_result(self, ensemble_id: str, article_id: str) -> Optional[Dict[str, Any]]:
        self.cursor.execute(
            """
            SELECT ensemble_id, article_id, extraction_text
            FROM extraction_results
            WHERE ensemble_id = ? AND article_id = ?
            """,
            (ensemble_id, article_id)
        )
        row = self.cursor.fetchone()
        if not row:
            return None

        return {
            "ensemble_id": row["ensemble_id"],
            "article_id": row["article_id"],
            "extraction_text": row["extraction_text"]
        }

    def retrieve_batch_info(self, ensemble_id: str) -> Optional[Dict[str, Any]]:
        self.cursor.execute("""
            SELECT
                ensemble_id,
                batch_ids,
                model,
                retrieval_status,
                variables,
                doc_ids,
                dataset_path,
                preprompt,
                prompt
            FROM batch_info
            WHERE ensemble_id = ?
        """, (ensemble_id,))
        row = self.cursor.fetchone()
        if not row:
            return None

        variable_dicts = json.loads(row["variables"]) if row["variables"] else []
        variable_objects = [VariableDefinition(**d) for d in variable_dicts]

        return {
            "ensemble_id": row["ensemble_id"],
            "batch_ids": row["batch_ids"].split(",") if row["batch_ids"] else [],
            "model": row["model"],
            "retrieval_status": row["retrieval_status"],
            "variables": variable_objects,
            "doc_ids": row["doc_ids"].split(",") if row["doc_ids"] else [],
            "dataset_path": row["dataset_path"],
            "preprompt": row["preprompt"],
            "prompt": row["prompt"]
        }

    def retrieve_variable_extractions(self, ensemble_id: str, article_id: str) -> Dict[str, str]:
        self.cursor.execute("""
            SELECT variable_name, extracted_value
            FROM extracted_variables
            WHERE ensemble_id = ? AND article_id = ?
        """, (ensemble_id, article_id))
        rows = self.cursor.fetchall()
        # Return a dict: { variable_name: extracted_value, ... }
        return {row["variable_name"]: row["extracted_value"] for row in rows}

    def mark_retrieved(self, ensemble_id: str):
        """
        Mark retrieval_status as 'done' for a given ensemble_id
        so that repeated retrieval calls won't re-download the same data.
        """
        self.cursor.execute(
            "UPDATE batch_info SET retrieval_status = 'done' WHERE ensemble_id = ?",
            (ensemble_id,)
        )
        self.conn.commit()

    def is_retrieval_done(self, ensemble_id: str) -> bool:
        """
        Returns True if retrieval_status is 'done' for the given ensemble_id;
        else returns False.
        """
        self.cursor.execute("""
            SELECT retrieval_status
            FROM batch_info
            WHERE ensemble_id = ?
        """, (ensemble_id,))
        row = self.cursor.fetchone()
        # Check retrieval_status, not ensemble_id
        return (row is not None) and (row["retrieval_status"] == 'done')