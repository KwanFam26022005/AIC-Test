from __future__ import annotations

import json
from pathlib import Path


def bulk_index_documents(cfg, docs_path: str | Path) -> int:
    from elasticsearch import Elasticsearch, helpers

    es = Elasticsearch(list(cfg.es.hosts), request_timeout=int(cfg.es.request_timeout))
    actions = []
    count = 0
    with Path(docs_path).open("r", encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            actions.append({"_index": cfg.es.index_name, "_id": f"{doc['video_id']}:{doc['frame_id']}", "_source": doc})
            if len(actions) >= int(cfg.es.bulk_size):
                ok, _ = helpers.bulk(es, actions)
                count += ok
                actions = []
    if actions:
        ok, _ = helpers.bulk(es, actions)
        count += ok
    return count

