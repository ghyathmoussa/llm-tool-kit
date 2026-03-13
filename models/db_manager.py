"""
Database Manager for PostgreSQL operations.
Handles connection pooling, transactions, and CRUD operations.
"""

import uuid
from contextlib import contextmanager
from typing import List, Dict, Optional, Generator, Set, Tuple

import psycopg2
from psycopg2 import pool, sql
from psycopg2.extras import execute_values

from utils.logger import setup_app_logger

logger = setup_app_logger(__name__)


class PostgreSQLManager:
    """Manages PostgreSQL connections and operations."""

    def __init__(self, database_url: str):
        """
        Initialize connection pool.

        Args:
            database_url: PostgreSQL connection URL (e.g., postgresql://user:pass@host:port/db)
        """
        self.database_url = database_url
        self.connection_pool = None

    def initialize_pool(self, min_connections: int = 1, max_connections: int = 10):
        """Initialize the connection pool."""
        try:
            self.connection_pool = pool.ThreadedConnectionPool(
                min_connections,
                max_connections,
                self.database_url
            )
            logger.info(f"Connection pool initialized: {min_connections}-{max_connections} connections")
        except Exception as e:
            logger.error(f"Failed to initialize connection pool: {e}")
            raise

    def close_pool(self):
        """Close all connections in the pool."""
        if self.connection_pool:
            self.connection_pool.closeall()
            logger.info("Connection pool closed")

    @contextmanager
    def get_connection(self):
        """Context manager for getting a connection from the pool."""
        if not self.connection_pool:
            self.initialize_pool()

        conn = None
        try:
            conn = self.connection_pool.getconn()
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Database operation failed: {e}")
            raise
        finally:
            if conn:
                self.connection_pool.putconn(conn)

    @contextmanager
    def get_cursor(self) -> Generator[Tuple, None, None]:
        """Context manager for getting a cursor with automatic cleanup."""
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                yield cursor, conn

    def get_existing_fatwa_chunks(self) -> Set[Tuple[int, int]]:
        """
        Get set of (fatwa_id, chunk_index) tuples already in output table.

        Returns:
            Set of tuples (fatwa_id, chunk_index)
        """
        query = """
            SELECT fatwa_id, chunk_index
            FROM fatwa_answer_chunks
        """
        with self.get_cursor() as (cursor, conn):
            cursor.execute(query)
            return set((row[0], row[1]) for row in cursor.fetchall())

    def get_existing_fatwa_ids(self) -> Set[int]:
        """
        Get set of fatwa_ids that have at least one chunk.

        Returns:
            Set of fatwa_ids
        """
        query = """
            SELECT DISTINCT fatwa_id
            FROM fatwa_answer_chunks
        """
        with self.get_cursor() as (cursor, conn):
            cursor.execute(query)
            return set(row[0] for row in cursor.fetchall())

    def delete_chunks_for_fatwa(self, fatwa_id: int):
        """
        Delete all chunks for a given fatwa_id.

        Args:
            fatwa_id: The fatwa ID to delete chunks for
        """
        query = sql.SQL("""
            DELETE FROM fatwa_answer_chunks
            WHERE fatwa_id = %s
        """)
        with self.get_cursor() as (cursor, conn):
            cursor.execute(query, (fatwa_id,))
            deleted_count = cursor.rowcount
            conn.commit()
            logger.info(f"Deleted {deleted_count} chunks for fatwa_id={fatwa_id}")

    def get_fatwas_to_process(
        self,
        text_column: str,
        limit: Optional[int] = None,
        skip_existing: bool = True,
        existing_fatwa_ids: Optional[Set[int]] = None
    ) -> Generator[Dict, None, None]:
        """
        Stream fatwas from the database for processing.

        Args:
            text_column: Name of column containing text to chunk
            limit: Optional limit on number of records to fetch
            skip_existing: Skip fatwas that already have chunks
            existing_fatwa_ids: Set of existing fatwa_ids to skip

        Yields:
            Dict with 'id' and 'text' keys
        """
        # Build the base query
        if skip_existing and existing_fatwa_ids:
            query = sql.SQL("""
                SELECT id, {text_col}
                FROM fatwas
                WHERE {text_col} IS NOT NULL AND {text_col} != ''
                AND id NOT IN %s
            """).format(text_col=sql.Identifier(text_column))
        else:
            query = sql.SQL("""
                SELECT id, {text_col}
                FROM fatwas
                WHERE {text_col} IS NOT NULL AND {text_col} != ''
            """).format(text_col=sql.Identifier(text_column))

        # Add limit if specified
        if limit:
            query = sql.SQL("{query} LIMIT %s").format(query=query)

        with self.get_cursor() as (cursor, conn):
            # Execute with appropriate parameters
            if limit:
                if skip_existing and existing_fatwa_ids:
                    cursor.execute(query, (tuple(existing_fatwa_ids), limit))
                else:
                    cursor.execute(query, (limit,))
            elif skip_existing and existing_fatwa_ids:
                cursor.execute(query, (tuple(existing_fatwa_ids),))
            else:
                cursor.execute(query)

            # Yield rows as dictionaries
            for row in cursor:
                yield {'id': row[0], 'text': row[1]}

    def insert_chunks_batch(self, chunks: List[Dict]) -> Tuple[int, int]:
        """
        Insert multiple chunks in a single transaction.

        Args:
            chunks: List of chunk dictionaries with keys:
                - fatwa_id: int
                - chunk_index: int
                - chunk_text: str
                - word_count: int
                - embedding_status: str (default: 'pending')

        Returns:
            Tuple of (attempted_count, inserted_count). Inserted count excludes duplicates skipped.
        """
        if not chunks:
            return 0, 0

        # Build the query with SQL identifiers for safety
        query = sql.SQL("""
            INSERT INTO fatwa_answer_chunks
                (id, fatwa_id, chunk_index, chunk_text, word_count, embedding_status, created_at, updated_at)
            VALUES {values}
            ON CONFLICT (fatwa_id, chunk_index) DO NOTHING
            RETURNING id
        """)

        # Build values clause with parameterized placeholders
        values_clauses = []
        params = []
        for chunk in chunks:
            chunk_uuid = str(uuid.uuid4())
            values_clauses.append(sql.SQL("(%s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"))
            params.extend([
                chunk_uuid,
                chunk['fatwa_id'],
                chunk['chunk_index'],
                chunk['chunk_text'],
                chunk['word_count'],
                chunk.get('embedding_status', 'pending')
            ])

        values_clause = sql.SQL(', ').join(values_clauses)
        query = sql.SQL("{query}").format(query=query)

        # Replace placeholder with values clause
        final_query = str(query).replace('{values}', str(values_clause))

        try:
            with self.get_cursor() as (cursor, conn):
                cursor.execute(final_query, params)
                inserted_count = len(cursor.fetchall())
                conn.commit()
                return len(chunks), inserted_count
        except Exception as e:
            logger.error(f"Failed to insert batch of {len(chunks)} chunks: {e}")
            raise
