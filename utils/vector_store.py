

from typing import Union
from llama_index.vector_stores.postgres import PGVectorStore
import sqlalchemy

class CustomVectorStore(PGVectorStore):
    def __init__(
        self,
        connection_string: Union[str, sqlalchemy.engine.URL],
        async_connection_string: Union[str, sqlalchemy.engine.URL],
        table_name: str,
        schema_name: str,
        hybrid_search: bool = False,
        text_search_config: str = "english",
        embed_dim: int = 1536,
        cache_ok: bool = False,
        perform_setup: bool = True,
        debug: bool = False,
        use_jsonb: bool = False,
    ) -> None:
        super().__init__(
            connection_string=connection_string,
            async_connection_string=async_connection_string,
            table_name=table_name,
            schema_name=schema_name,
            hybrid_search=hybrid_search,
            text_search_config=text_search_config,
            embed_dim=embed_dim,
            cache_ok=cache_ok,
            perform_setup=perform_setup,
            debug=debug,
            use_jsonb=use_jsonb,
        )

    def _create_bm25_index_if_not_exists(self)->None:
        """create index for embedding and bm25 search"""
        with self._session() as session, session.begin():
            statement = sqlalchemy.text(f"SELECT COUNT(1) FROM PG_INDEXES WHERE STARTS_WITH(indexname,'{self.table_name}_bm25_index')")
            result = session.execute(statement)
            row = result.fetchone()
            if row[0]>0:
                session.commit()
                return
            SQL_CREATE_TEXT_INDEX = """
            CALL paradedb.create_bm25(
                    index_name => '{index_name}',
                    table_name => '{table_name}',
                    key_field => 'id',
                    text_fields => '{{text: {{tokenizer: {{type: "chinese_lindera"}}}}}}'
            );
            """
            statement = sqlalchemy.text(SQL_CREATE_TEXT_INDEX.format(
                index_name=self.table_name,
                table_name=f"data_{self.table_name}"
            ))
            session.execute(statement)
            session.commit()
            
    def _initialize(self) -> None:
        if not self._is_initialized:
            self._connect()
            if self.perform_setup:
                self._create_extension()
                self._create_schema_if_not_exists()
                self._create_tables_if_not_exists()
                self._create_bm25_index_if_not_exists()
            self._is_initialized = True
