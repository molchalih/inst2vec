import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from modules.database import UserCluster


def test_user_cluster_columns():
    cols = {c.key for c in UserCluster.__table__.columns}
    assert "user_pk" in cols
    assert "embedding_case" in cols
    assert "cluster_id" in cols
    assert "umap_x" in cols
    assert "umap_y" in cols
    assert "created_at" in cols
    assert "updated_at" in cols
    assert UserCluster.__tablename__ == "user_clusters"
