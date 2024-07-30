
from contextlib import contextmanager
import json
from typing import List, Tuple
from llama_index.core.retrievers import BaseRetriever
from sqlalchemy import make_url
import psycopg2
from llama_index.core.schema import NodeWithScore,TextNode
from psycopg2 import pool

import os
import time
import math
import json
import random
import requests
import numpy as np
from retry import retry
from tqdm import tqdm


class CustomBM25Retriever(BaseRetriever):
    """Custom retriever for pgvector with bm25"""

    def __init__(self,connection_string:str,table_name:str, index_name:str=None,top_k:int=5, keywords_search:bool=True, **kwargs):
        super().__init__(**kwargs)
        # dialect+driver://username:password@host:port/database
        self.url = make_url(connection_string)
        self.pool = pool.SimpleConnectionPool(
            1,
            5,
            host=self.url.host,
            port=self.url.port,
            user=self.url.username,
            password=self.url.password,
            database=self.url.database,
        )
        self.keywords_search=keywords_search
        self.top_k = top_k
        self.table_name = table_name
        if not index_name:
            self.index_name = table_name
        else:
            self.index_name = index_name

    @contextmanager
    def _get_cursor(self):
        conn = self.pool.getconn()
        cur = conn.cursor()
        try:
            yield cur
        finally:
            cur.close()
            conn.commit()
            self.pool.putconn(conn)

    def _retrieve(self, query, **kwargs):
        # Implement your custom BM25 retrieval logic here
        # This method should return a list of nodes (documents) that are relevant to the query
        result = []
        if self.keywords_search:
            with self._get_cursor() as cur:
                # Execute your custom BM25 query here
                full_text_search_sql = """
                SELECT  t.text,t.metadata_ as meta, s.score
                FROM {table_name} t
                RIGHT JOIN(
                    SELECT id, paradedb.rank_bm25(id) as score
                    FROM {idx_name}.search(
                        query => paradedb.term_set(terms => ARRAY[{terms}]), limit_rows => {top_k}
                    )
                ) s
                ON t.id = s.id;
                """
                query = query.query_str.split(' ')
                terms = ', '.join([f"paradedb.term(field => 'text', value => '{term}')" for term in query])
                cur.execute(full_text_search_sql.format(table_name=self.table_name, idx_name=self.index_name,top_k=self.top_k, terms=terms))
                for record in cur:
                    text, meta,score, = record
                    _node_content = meta.get('_node_content')
                    content_dic = json.loads(_node_content)
                    node_with_score = NodeWithScore(node=TextNode(text=text, id_=content_dic.get('id_')), score=score)
                    result.append(node_with_score)
                return result if len(result)>0 else None
        else:
            with self._get_cursor() as cur:
                # Execute your custom BM25 query here
                full_text_search_sql = """
                SELECT  t.text,t.metadata_ as meta, s.score
                FROM {table_name} t
                RIGHT JOIN(
                    SELECT id, paradedb.rank_bm25(id) as score
                    FROM {idx_name}.search(
                        query => paradedb.parse('text:{query}'), limit_rows => {top_k}
                    )
                ) s
                ON t.id = s.id;
                """
                cur.execute(full_text_search_sql.format(table_name=self.table_name, idx_name=self.index_name,top_k=self.top_k, query=query))
                for record in cur:
                    text, meta,score, = record
                    _node_content = meta.get('_node_content')
                    content_dic = json.loads(_node_content)
                    node_with_score = NodeWithScore(node=TextNode(text=text, id_=content_dic.get('id_')), score=score)
                    result.append(node_with_score)
                if len(result)==0:
                    # 防止检索不到结果报错
                    fake_node = NodeWithScore(node=TextNode(text='text', id_='fake_id'), score=0.0)
                    result.append(fake_node)
                return result

class CustomEmbeddingRetriever(BaseRetriever):
    """Custom retriever for pgvector embedding"""

    def __init__(self,connection_string:str,table_name:str,top_k:int=5, **kwargs):
        super().__init__(**kwargs)
        # dialect+driver://username:password@host:port/database
        self.url = make_url(connection_string)
        # self.conn = psycopg2.connect(
        #     dbname=self.url.database,
        #     user=self.url.username,
        #     password=self.url.password,
        #     host=self.url.host,
        #     port=self.url.port
        # )
        self.pool = pool.SimpleConnectionPool(
            1,
            5,
            host=self.url.host,
            port=self.url.port,
            user=self.url.username,
            password=self.url.password,
            database=self.url.database,
        )
        self.top_k = top_k
        self.table_name = table_name
        self.embedding_query_dic = EmbeddingCache().load()[0]

    @contextmanager
    def _get_cursor(self):
        conn = self.pool.getconn()
        cur = conn.cursor()
        try:
            yield cur
        finally:
            cur.close()
            conn.commit()
            self.pool.putconn(conn)

    def _retrieve(self, query, **kwargs):
        # Implement your custom embedding retrieval logic here
        # This method should return a list of nodes (documents) that are relevant to the query
        result = []
        if query.query_str in self.embedding_query_dic:
            embedding_query = self.embedding_query_dic[query.query_str]
        else:
            embedding_query = EmbeddingCache().get_ollama_embedding(query.query_str)
            self.embedding_query_dic[query.query_str] = embedding_query
        with self._get_cursor() as cur:
            cur.execute(
                f"SELECT metadata_, text, embedding <=> %s AS distance FROM {self.table_name} ORDER BY distance LIMIT {self.top_k}",
                (json.dumps(embedding_query),),
            )
            # score_threshold = kwargs.get("score_threshold") if kwargs.get("score_threshold") else 0.0
            for record in cur:
                metadata, text, distance = record
                score = 1 - distance
                
                _node_content = metadata.get('_node_content')
                content_dic = json.loads(_node_content)
                node_with_score = NodeWithScore(node=TextNode(text=text, id_=content_dic.get('id_')), score=score)
                result.append(node_with_score)
                
                # if score > score_threshold:
                #     docs.append(Document(page_content=text, metadata=metadata))
        return result

class CustomEnsembleRetriever(BaseRetriever):
    """Custom retriever for pgvector with bm25 and embedding"""

    def __init__(self,connection_string:str,table_name:str, index_name:str=None,top_k:int=5, **kwargs):
        super().__init__(**kwargs)
        # dialect+driver://username:password@host:port/database
        self.url = make_url(connection_string)
        self.pool = pool.SimpleConnectionPool(
            1,
            5,
            host=self.url.host,
            port=self.url.port,
            user=self.url.username,
            password=self.url.password,
            database=self.url.database,
        )
        self.top_k = top_k
        self.table_name = table_name
        if not index_name:
            self.index_name = table_name
        else:
            self.index_name = index_name
        self.embedding_query_dic,self.jieba_query_dic = EmbeddingCache().load()

    @contextmanager
    def _get_cursor(self):
        conn = self.pool.getconn()
        cur = conn.cursor()
        try:
            yield cur
        finally:
            cur.close()
            conn.commit()
            self.pool.putconn(conn)

    def _retrieve(self, query, **kwargs):
        # Implement your custom BM25 retrieval logic here
        # This method should return a list of nodes (documents) that are relevant to the query
        result = []
        # bm25 key words
        with self._get_cursor() as cur:
            # Execute your custom BM25 query here
            full_text_search_sql = """
            SELECT  t.text,t.metadata_ as meta, s.score
            FROM {table_name} t
            RIGHT JOIN(
                SELECT id, paradedb.rank_bm25(id) as score
                FROM {idx_name}.search(
                    query => paradedb.term_set(terms => ARRAY[{terms}]), limit_rows => {top_k}
                )
            ) s
            ON t.id = s.id;
            """
            query_terms_str = self.jieba_query_dic[query.query_str]
            terms = ', '.join([f"paradedb.term(field => 'text', value => '{term}')" for term in query_terms_str.split(' ')])
            cur.execute(full_text_search_sql.format(table_name=self.table_name, idx_name=self.index_name,top_k=self.top_k, terms=terms))
            for record in cur:
                text, meta,score, = record
                _node_content = meta.get('_node_content')
                content_dic = json.loads(_node_content)
                node_with_score = NodeWithScore(node=TextNode(text=text, id_=content_dic.get('id_')), score=score)
                result.append(node_with_score)
        # embedding 
        with self._get_cursor() as cur:
            if query.query_str in self.embedding_query_dic:
                embedding_query = self.embedding_query_dic[query.query_str]
            else:
                embedding_query = EmbeddingCache().get_ollama_embedding(query.query_str)
                self.embedding_query_dic[query.query_str] = embedding_query
            with self._get_cursor() as cur:
                cur.execute(
                    f"SELECT metadata_, text, embedding <=> %s AS distance FROM {self.table_name} ORDER BY distance LIMIT {self.top_k}",
                    (json.dumps(embedding_query),),
                )
                # score_threshold = kwargs.get("score_threshold") if kwargs.get("score_threshold") else 0.0
                for record in cur:
                    metadata, text, distance = record
                    score = 1 - distance
                    
                    _node_content = metadata.get('_node_content')
                    content_dic = json.loads(_node_content)
                    node_with_score = NodeWithScore(node=TextNode(text=text, id_=content_dic.get('id_')), score=score)
                    result.append(node_with_score)
        # rerank
        reranker = XInferenceReranker(url="http://10.24.20.73", port=9997, model="bge-reranker-base")
        docs = [node.text for node in result]
        reranked_docs = reranker.rerank(query=query.query_str, docs=docs,top_n=self.top_k)
        rerank_result = []
        for doc in reranked_docs:
            idx = doc['index']
            rerank_result.append(result[idx])
        return rerank_result

class EmbeddingCache(object):
    def __init__(self):
        pass

    @staticmethod
    @retry(exceptions=Exception, tries=3, max_delay=20)
    def get_ollama_embedding(req_text: str):
        time.sleep(random.random() / 2)
        url = "http://10.24.20.73:11434/api/embeddings"
        headers = {'Content-Type': 'application/json'}
        payload = json.dumps({"model": "gte-Qwen2-1.5B-instruct:latest", "prompt": req_text})
        new_req = requests.request("POST", url, headers=headers, data=payload)
        resp_json = new_req.json()
        embedding = resp_json['embedding']
        return embedding

    @staticmethod
    @retry(exceptions=Exception, tries=3, max_delay=20)
    def get_openai_embedding(req_text: str):
        time.sleep(random.random() / 2)
        url = "https://api.openai.com/v1/embeddings"
        headers = {'Content-Type': 'application/json', "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"}
        payload = json.dumps({"model": "text-embedding-ada-002", "input": req_text})
        new_req = requests.request("POST", url, headers=headers, data=payload)
        return new_req.json()['data'][0]['embedding']

    @staticmethod
    @retry(exceptions=Exception, tries=3, max_delay=20)
    def get_bge_embedding(req_text: str):
        url = "http://localhost:50073/embedding"
        headers = {'Content-Type': 'application/json'}
        payload = json.dumps({"text": req_text})
        new_req = requests.request("POST", url, headers=headers, data=payload)
        return new_req.json()['embedding']

    @staticmethod
    @retry(exceptions=Exception, tries=3, max_delay=20)
    def get_jina_embedding(req_text: str):
        time.sleep(random.random() / 2)
        url = 'https://api.jina.ai/v1/embeddings'
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {os.getenv("JINA_API_KEY")}'
        }
        data = {
            'input': [req_text],
            'model': 'jina-embeddings-v2-base-zh'
        }
        response = requests.post(url, headers=headers, json=data)
        embedding = response.json()["data"][0]["embedding"]
        embedding_norm = math.sqrt(sum([i**2 for i in embedding]))
        return [i/embedding_norm for i in embedding]

    def build_with_context(self, context_type: str):
        with open("../data/doc_qa_test.json", "r", encoding="utf-8") as f:
            content = json.loads(f.read())
        queries = list(content[context_type].values())
        query_num = len(queries)
        embedding_data = np.empty(shape=[query_num, 768])
        for i in tqdm(range(query_num), desc="generate embedding"):
            embedding_data[i] = self.get_bge_embedding(queries[i])
        np.save(f"../data/{context_type}_bce_embedding.npy", embedding_data)

    def build(self):
        self.build_with_context("queries")
        self.build_with_context("corpus")

    @staticmethod
    def load(query_write=False):
        current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        queries_embedding_data = np.load(os.path.join(current_dir, "data/medical/tmp/query_embeddings.npy"))
        queries = np.load(os.path.join(current_dir, "data/medical/tmp/querys.npy"))
        jieba_queries = np.load(os.path.join(current_dir, "data/medical/tmp/jieba_ask.npy"))
        query_embedding_dict = {}
        jieba_queries_dict = {}
        # with open(os.path.join(current_dir, "data/doc_qa_test.json"), "r", encoding="utf-8") as f:
        #     content = json.loads(f.read())
        # queries = list(content["queries"].values())
        # corpus = list(content["corpus"].values())
        for i in range(len(queries)):
            query_embedding_dict[queries[i]] = queries_embedding_data[i].tolist()
            jieba_queries_dict[queries[i]] = jieba_queries[i]
        # if query_write:
        #     rewrite_queries_embedding_data = np.load(os.path.join(current_dir, "data/query_rewrite_openai_embedding.npy"))
        #     with open("../data/query_rewrite.json", "r", encoding="utf-8") as f:
        #         rewrite_content = json.loads(f.read())

        #     rewrite_queries_list = []
        #     for original_query, rewrite_queries in rewrite_content.items():
        #         rewrite_queries_list.extend(rewrite_queries)
        #     for i in range(len(rewrite_queries_list)):
        #         query_embedding_dict[rewrite_queries_list[i]] = rewrite_queries_embedding_data[i].tolist()
        return query_embedding_dict, jieba_queries_dict

class XInferenceReranker():
    def __init__(self, url:str,model:str,port) -> None:
        self.url = url
        self.port = port
        self.model = model

    def rerank(self, query: str, docs: List[str], top_n:int=5)->List[Tuple]:
        rerank_url = f"{self.url}:{self.port}/v1/rerank"
        payload = json.dumps({
        "query": query,
        "documents": docs,
        "top_n": top_n,
        "model":self.model
        })
        headers = {'Content-Type': 'application/json'}
        response = requests.request("POST", rerank_url, headers=headers, data=payload)
        resp_json = response.json()
        return resp_json['results']
        
        
if __name__ == '__main__':
    from pprint import pprint
    connection_string = "postgresql://postgres:postgres@localhost:5436/medical"
    # custom_bm25_retriever = CustomBM25Retriever(connection_string=connection_string, 
    #                                             table_name='data_medical_data_embedding_t1',
    #                                             index_name='medical_data_embedding_t1',
    #                                             top_k=3,
    #                                             keywords_search=False)
    # # query = "党参 高血压 泡水 您好 女婿 两天 时候 可以"
    # query = "我有高血压这两天女婿来的时候给我拿了些党参泡水喝，您好高血压可以吃党参吗？"
    # t_result = custom_bm25_retriever.retrieve(str_or_query_bundle=query)
    # pprint(t_result)
    
    # custom_retriever = CustomEmbeddingRetriever(connection_string=connection_string,
    #                                             table_name='data_medical_data_embedding_t1',
    #                                             top_k=3,)
    # query = "我有高血压这两天女婿来的时候给我拿了些党参泡水喝，您好高血压可以吃党参吗？"
    # t_result = custom_retriever.retrieve(str_or_query_bundle=query)
    # pprint(t_result)
    # dic = EmbeddingCache().load()[0]
    # print(len(dic))
    # res = dic['我有高血压这两天女婿来的时候给我拿了些党参泡水喝，您好高血压可以吃党参吗？']
    # print(res)
    
    reranker = XInferenceReranker(url="http://10.24.20.73", port=9997, model="bge-reranker-base")
    res = reranker.rerank(query="who is Bob", docs=["Bob is a lawer", "Tom is a labor"])
    pprint(res)