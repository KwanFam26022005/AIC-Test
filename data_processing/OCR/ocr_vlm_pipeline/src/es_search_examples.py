from __future__ import annotations


def fuzzy_ocr_query(query: str, size: int = 20) -> dict:
    return {
        "size": size,
        "_source": ["video_id", "frame_id", "timestamp_ms", "frame_path", "ocr.corrected_text"],
        "query": {
            "multi_match": {
                "query": query,
                "fields": ["ocr.corrected_text^3", "ocr.normalized_text^2", "ocr.raw_text"],
                "fuzziness": "AUTO",
            }
        },
    }


def search(cfg, query: str, size: int = 20):
    from elasticsearch import Elasticsearch

    es = Elasticsearch(list(cfg.es.hosts), request_timeout=int(cfg.es.request_timeout))
    return es.search(index=cfg.es.index_name, body=fuzzy_ocr_query(query, size=size))
