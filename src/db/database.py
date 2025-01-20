import sqlite3
from typing import List, Dict, Any, Optional
import json
from src.api.data_extraction import VariableDefinition

class Database:
    def __init__(self, path="data_extraction3.db"):
        # Use Row objects to allow row["column_name"] access
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.cursor = self.conn.cursor()
        self._create_tables()

    def _create_tables(self):
        # Store batch processing information
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS batch_info (
                prompt_run_id TEXT PRIMARY KEY,
                sub_ensemble_id TEXT,
                ensemble_id TEXT,
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
                prompt_run_id TEXT,
                sub_ensemble_id TEXT,
                ensemble_id TEXT,
                article_id TEXT,
                extraction_text TEXT,
                PRIMARY KEY (prompt_run_id, article_id)
            )
        """)
        
        # Store variable extractions
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS extracted_variables (
                prompt_run_id TEXT,
                sub_ensemble_id TEXT,
                ensemble_id TEXT,
                article_id TEXT,
                variable_name TEXT,
                extracted_value TEXT,
                PRIMARY KEY (prompt_run_id, article_id, variable_name)
            )
        """)
        
        # Store master variables
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS ensemble_vars (
                ensemble_id TEXT PRIMARY KEY,
                variables_json TEXT
            )
        """)
        
        self.conn.commit()

    def store_extraction_result(
            self,
            prompt_run_id: str,
            sub_ensemble_id: str,
            ensemble_id: str,
            article_id: str,
            extraction_text: str
        ):
            self.cursor.execute("""
                INSERT OR REPLACE INTO extraction_results (
                    prompt_run_id,
                    sub_ensemble_id,
                    ensemble_id,
                    article_id,
                    extraction_text
                ) VALUES (?, ?, ?, ?, ?)
            """, (prompt_run_id, sub_ensemble_id, ensemble_id, article_id, extraction_text))
            self.conn.commit()

    def store_multiple_extraction_results(
        self,
        prompt_run_id: str,
        sub_ensemble_id: str,
        ensemble_id: str,
        article_ids: List[str],
        extraction_texts: List[str]
    ):
        data = []
        for doc_id, text in zip(article_ids, extraction_texts):
            data.append((prompt_run_id, sub_ensemble_id, ensemble_id, doc_id, text))

        self.cursor.executemany("""
            INSERT OR REPLACE INTO extraction_results (
                prompt_run_id,
                sub_ensemble_id,
                ensemble_id,
                article_id,
                extraction_text
            ) VALUES (?, ?, ?, ?, ?)
        """, data)
        self.conn.commit()

    def store_batch_info(
        self,
        prompt_run_id: str,
        sub_ensemble_id: str,
        ensemble_id: str,
        batch_ids: List[str],
        model: str,
        variables: List[VariableDefinition],
        doc_ids: List[str],
        dataset_path: str,
        preprompt: str,
        prompt: str
    ) -> None:
        # Convert variables to JSON; store metadata for a single prompt_run_id (lowest level)
        variables_json = json.dumps([v.dict() for v in variables])
        # Join batch_ids and doc_ids into comma-separated for storage
        batch_ids_str = ",".join(batch_ids)
        doc_ids_str = ",".join(doc_ids) if doc_ids else ""

        self.cursor.execute("""
            INSERT INTO batch_info (
                prompt_run_id,
                sub_ensemble_id,
                ensemble_id,
                batch_ids,
                model,
                retrieval_status,
                variables,
                doc_ids,
                dataset_path,
                preprompt,
                prompt
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            prompt_run_id,
            sub_ensemble_id,
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

    def store_variable_extraction(
        self,
        prompt_run_id: str,
        sub_ensemble_id: str,
        ensemble_id: str,
        article_id: str,
        variable_name: str,
        extracted_value: str
    ):
        self.cursor.execute(
            """
            INSERT OR REPLACE INTO extracted_variables (
                prompt_run_id,
                sub_ensemble_id,
                ensemble_id,
                article_id,
                variable_name,
                extracted_value
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (prompt_run_id, sub_ensemble_id, ensemble_id, article_id, variable_name, extracted_value)
        )
        self.conn.commit()

    def store_multiple_variable_extractions(
        self,
        prompt_run_id: str,
        sub_ensemble_id: str,
        ensemble_id: str,
        article_id: str,
        extractions: Dict[str, Any]
    ):
        data = []
        for var_name, value in extractions.items():
            if isinstance(value, list):
                value_str = ", ".join(str(v) for v in value)
            else:
                value_str = str(value)
            data.append((
                prompt_run_id,
                sub_ensemble_id,
                ensemble_id,
                article_id,
                var_name,
                value_str
            ))

        self.cursor.executemany("""
            INSERT OR REPLACE INTO extracted_variables (
                prompt_run_id,
                sub_ensemble_id,
                ensemble_id,
                article_id,
                variable_name,
                extracted_value
            ) VALUES (?, ?, ?, ?, ?, ?)
        """, data)
        self.conn.commit()

    def retrieve_extraction_result(
        self,
        prompt_run_id: str,
        article_id: str
    ) -> Optional[Dict[str, Any]]:
        self.cursor.execute("""
            SELECT prompt_run_id, sub_ensemble_id, ensemble_id, article_id, extraction_text
            FROM extraction_results
            WHERE prompt_run_id = ? AND article_id = ?
        """, (prompt_run_id, article_id))
        row = self.cursor.fetchone()
        if not row:
            return None
        return {
            "prompt_run_id": row["prompt_run_id"],
            "sub_ensemble_id": row["sub_ensemble_id"],
            "ensemble_id": row["ensemble_id"],
            "article_id": row["article_id"],
            "extraction_text": row["extraction_text"]
        }

    def retrieve_batch_info(self, prompt_run_id: str) -> Optional[Dict[str, Any]]:
        self.cursor.execute("""
            SELECT
                prompt_run_id,
                sub_ensemble_id,
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
            WHERE prompt_run_id = ?
        """, (prompt_run_id,))
        row = self.cursor.fetchone()
        if not row:
            return None

        variable_dicts = json.loads(row["variables"]) if row["variables"] else []
        variable_objects = [VariableDefinition(**d) for d in variable_dicts]

        return {
            "prompt_run_id": row["prompt_run_id"],            
            "sub_ensemble_id": row["sub_ensemble_id"],
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

    def retrieve_variable_extractions(self, prompt_run_id: str, article_id: str) -> Dict[str, str]:
        self.cursor.execute("""
            SELECT variable_name, extracted_value
            FROM extracted_variables
            WHERE prompt_run_id = ? AND article_id = ?
        """, (prompt_run_id, article_id))
        rows = self.cursor.fetchall()
        # Return a dict: { variable_name: extracted_value, ... }
        return {row["variable_name"]: row["extracted_value"] for row in rows}

    def mark_retrieved(self, prompt_run_id: str):
        """
        Mark retrieval_status as 'done' for a given prompt_run_id
        so that repeated retrieval calls won't re-download the same data.
        """
        self.cursor.execute(
            "UPDATE batch_info SET retrieval_status = 'done' WHERE prompt_run_id = ?",
            (prompt_run_id,)
        )
        self.conn.commit()

    def is_retrieval_done(self, prompt_run_id: str) -> bool:
        """
        Returns True if retrieval_status is 'done' for the given prompt_run_id;
        else returns False.
        """
        self.cursor.execute("""
            SELECT retrieval_status
            FROM batch_info
            WHERE prompt_run_id = ?
        """, (prompt_run_id,))
        row = self.cursor.fetchone()
        # Check retrieval_status, not prompt_run_id
        return (row is not None) and (row["retrieval_status"] == 'done')
    
    def find_prompt_runs_for_ensemble(self, ensemble_id: str) -> List[str]:
        """
        Return a list of prompt_run_ids for a given ensemble_id.
        """
        self.cursor.execute("""
            SELECT prompt_run_id
            FROM batch_info
            WHERE ensemble_id = ?
        """, (ensemble_id,))
        rows = self.cursor.fetchall()
        return [r["prompt_run_id"] for r in rows]

    def store_master_variables(self, ensemble_id: str, variables_json: str):
        self.cursor.execute("""
            INSERT OR REPLACE INTO ensemble_vars (ensemble_id, variables_json)
            VALUES (?, ?)
        """, (ensemble_id, variables_json))
        self.conn.commit()

    def retrieve_master_variables(self, ensemble_id: str) -> Optional[List[Dict[str, Any]]]:
        self.cursor.execute("""
            SELECT variables_json
            FROM ensemble_vars
            WHERE ensemble_id = ?
        """, (ensemble_id,))
        row = self.cursor.fetchone()
        if not row:
            return None
        return json.loads(row["variables_json"])