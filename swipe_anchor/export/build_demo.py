"""Populate the app store with REAL creators from the pipeline DB (plan §6, §5.3).

Reads the pipeline main DB **read-only** (the decoupled fusion-script pattern: a
plain ``create_engine`` + SELECTs only — no import of ``core``) and writes the
swipe-anchor app store: creators, digests (rep clips + caption/audio cues), the
local media URLs, and within-cluster confusable comparisons.

This is the pragmatic "show real content" build. Representative clips are picked
by play-count (a neutral popularity rule); the standardized-medoid bias guard
(`core/selection.py`) is the eventual upgrade for the real anchor run.

    APP_DATABASE_URL=sqlite:///data/swipe_anchor.db \
    uv run python -m swipe_anchor.export.build_demo \
        --pipeline-db ~/inst2vec/data/inst2vec.db --case sandwich --media-base /media
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from collections import Counter

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from swipe_anchor.db import create_app_engine
from swipe_anchor.db.models import (
    Assignment,
    Comparison,
    Consensus,
    Creator,
    CreatorDigest,
    DigestClip,
    GoldItem,
    ReliabilityEvent,
    Response,
    Triplet,
)

_STOP = set(
    ["the", "a", "an", "and", "or", "of", "to", "in", "for", "with", "on", "at", "is", "are", "be", "was", "were", "this", "that", "it", "as", "by", "you", "your", "our", "we", "us", "my", "me", "his", "her", "their", "they", "them", "from", "out", "about", "into", "over", "new", "all", "not", "but", "can", "will", "just", "get", "got", "how", "why", "what", "when", "who", "not", "via", "more", "most"]
)


def _keywords(texts: list[str | None]) -> list[str]:
    tokens: list[str] = []
    tags: list[str] = []
    for t in texts:
        if not t:
            continue
        tags += [h.lower() for h in re.findall(r"#(\w+)", t)]
        tokens += [w for w in re.findall(r"[a-z]{3,}", t.lower()) if w not in _STOP]
    top_tags = [f"#{h}" for h, _ in Counter(tags).most_common(4)]
    top_words = [w for w, _ in Counter(tokens).most_common(6)]
    return top_tags + top_words


def _split_labels(value: str | None) -> list[str]:
    return [x.strip() for x in (value or "").split(",") if x.strip()]


def make_gold_items(
    creators: list[dict], rng: random.Random, max_gold: int
) -> list[dict]:
    """Auto gold: two creators from one group + one from a different group.

    ``known_odd`` is the out-group creator — an obvious odd-one-out (plan §7.1).
    Deterministic for a fixed ``rng``. Returns dicts ready to write as a
    ``status='gold'`` Comparison + a GoldItem row.
    """
    by_group: dict[str, list[dict]] = {}
    for c in creators:
        by_group.setdefault(c["group"], []).append(c)
    groups = sorted(g for g, m in by_group.items() if len(m) >= 2)
    out: list[dict] = []
    for i, g in enumerate(groups):
        others = [og for og in by_group if og != g and by_group[og]]
        if not others:
            continue
        pair = sorted(by_group[g], key=lambda c: c["id"])[:2]
        odd = rng.choice(by_group[rng.choice(sorted(others))])
        trio = sorted([pair[0]["id"], pair[1]["id"], odd["id"]])
        out.append(
            {
                "comparison_id": f"gold-{g}-{i}",
                "creator_a": trio[0],
                "creator_b": trio[1],
                "creator_c": trio[2],
                "known_odd": odd["id"],
                "seed_group": g,
            }
        )
        if len(out) >= max_gold:
            break
    return out


def make_boundary_comparisons(
    creators: list[dict], rng: random.Random, max_boundary: int
) -> list[dict]:
    """Boundary triples: two creators from cluster X + one from adjacent cluster Y."""
    by_cluster: dict[int, list[dict]] = {}
    for c in creators:
        by_cluster.setdefault(c["cluster"], []).append(c)
    clusters = sorted(k for k, m in by_cluster.items() if len(m) >= 2)
    out: list[dict] = []
    for i in range(len(clusters)):
        x = clusters[i]
        y = clusters[(i + 1) % len(clusters)]
        if x == y or not by_cluster[y]:
            continue
        pair = sorted(by_cluster[x], key=lambda c: c["id"])[:2]
        other = sorted(by_cluster[y], key=lambda c: c["id"])[0]
        trio = sorted([pair[0]["id"], pair[1]["id"], other["id"]])
        out.append(
            {
                "comparison_id": f"bound-{x}-{y}-{i}",
                "creator_a": trio[0],
                "creator_b": trio[1],
                "creator_c": trio[2],
                "kind": "boundary",
                "seed_group": pair[0]["group"],
            }
        )
        if len(out) >= max_boundary:
            break
    return out


def _build(args: argparse.Namespace) -> None:
    pipe = create_engine(f"sqlite:///{os.path.expanduser(args.pipeline_db)}")
    app_engine = create_app_engine(
        os.environ.get("APP_DATABASE_URL") or "sqlite:///data/swipe_anchor.db"
    )

    with pipe.connect() as pc:
        # cluster_id -> human label (from the LLM cluster summary payload)
        labels: dict[int, str] = {}
        for cid, payload in pc.execute(
            text(
                "select cluster_id, payload from cluster_labels "
                "where embedding_case=:c and payload is not null"
            ),
            {"c": args.case},
        ):
            try:
                labels[int(cid)] = (
                    json.loads(payload).get("cluster_label") or f"cluster {cid}"
                )
            except (ValueError, TypeError):
                labels[int(cid)] = f"cluster {cid}"

        # eligible users: clustered (non-noise) with at least one downloaded clip
        rows = pc.execute(
            text(
                "select uc.user_id, uc.cluster_id "
                "from user_clusters uc "
                "where uc.embedding_case=:c and uc.cluster_id>=0 "
                "and exists (select 1 from clips cl where cl.user_id=uc.user_id "
                "            and cl.is_downloaded=1)"
            ),
            {"c": args.case},
        ).all()

        creators: list[dict] = []
        for uid, cid in rows:
            clips = pc.execute(
                text(
                    "select id, caption_clean, caption_translation "
                    "from clips where user_id=:u and is_downloaded=1 "
                    "order by coalesce(play_count,0) desc limit :n"
                ),
                {"u": uid, "n": args.clips_per_user},
            ).all()
            if not clips:
                continue
            clip_ids = [c[0] for c in clips]
            caps = _keywords([c[2] or c[1] for c in clips])
            audio_rows = pc.execute(
                text(
                    "select genre_labels, moodtheme_labels, instrument_labels "
                    "from audio_mir where clip_id in "
                    f"({','.join(str(i) for i in clip_ids)})"
                )
            ).all()
            genre: list[str] = []
            mood: list[str] = []
            instr: list[str] = []
            for g, m, i in audio_rows:
                genre += _split_labels(g)
                mood += _split_labels(m)
                instr += _split_labels(i)
            n_total = pc.execute(
                text("select count(*) from clips where user_id=:u and is_downloaded=1"),
                {"u": uid},
            ).scalar_one()
            creators.append(
                {
                    "id": int(uid),
                    "cluster": int(cid),
                    "group": labels.get(int(cid), f"cluster {cid}"),
                    "clip_ids": clip_ids,
                    "captions": caps,
                    "audio": {
                        "genre_labels": [x for x, _ in Counter(genre).most_common(4)],
                        "moodtheme_labels": [
                            x for x, _ in Counter(mood).most_common(4)
                        ],
                        "instrument_labels": [
                            x for x, _ in Counter(instr).most_common(4)
                        ],
                    },
                    "n_clips": int(n_total),
                }
            )

    _write_app_store(app_engine, creators, args)
    print(
        f"built {len(creators)} creators from case={args.case!r}; "
        f"comparisons written (see app store)"
    )


def reset_content_tables(s: Session) -> None:
    """Delete content tables in FK-safe order; keep annotators + access_codes.

    ``ReliabilityEvent`` FKs ``comparisons``, so it MUST be cleared before them or
    the comparison delete raises an IntegrityError under SQLite ``foreign_keys=ON``
    once any answers have been collected (re-running this seeder on a live store).
    It also FKs ``annotators``, which are kept — deleting the events leaves the
    annotator rows intact.
    """
    for model in (
        Triplet,
        ReliabilityEvent,
        Response,
        Assignment,
        Consensus,
        GoldItem,
        Comparison,
        DigestClip,
        CreatorDigest,
        Creator,
    ):
        s.query(model).delete()
    s.flush()


def _write_app_store(
    app_engine, creators: list[dict], args: argparse.Namespace
) -> None:
    import random

    rng = random.Random(0)
    with Session(app_engine) as s:
        reset_content_tables(s)

        for c in creators:
            s.add(
                Creator(
                    creator_id=c["id"],
                    seed_cluster_id=c["cluster"],
                    seed_group=c["group"],
                )
            )
            s.add(
                CreatorDigest(
                    creator_id=c["id"],
                    digest_version=1,
                    rep_clip_ids=c["clip_ids"],
                    caption_keywords={"keywords": c["captions"]},
                    audio_summary=c["audio"],
                    n_clips=c["n_clips"],
                )
            )
            for ord_, clip_id in enumerate(c["clip_ids"]):
                s.add(
                    DigestClip(
                        creator_id=c["id"],
                        clip_id=clip_id,
                        ord=ord_,
                        video_url=f"{args.media_base}/videos/{clip_id}.mp4",
                        poster_url=f"{args.media_base}/thumbnails/{clip_id}.jpg",
                        is_medoid=(ord_ == 0),
                    )
                )

        # Within-cluster confusable comparisons: chunk each cluster's shuffled
        # members into triples; two passes for more coverage. Capped overall.
        by_cluster: dict[int, list[dict]] = {}
        for c in creators:
            by_cluster.setdefault(c["cluster"], []).append(c)

        n = 0
        for _pass in range(2):
            for cid, members in by_cluster.items():
                if len(members) < 3:
                    continue
                pool = members[:]
                rng.shuffle(pool)
                for i in range(0, len(pool) - 2, 3):
                    a, b, d = pool[i], pool[i + 1], pool[i + 2]
                    s.add(
                        Comparison(
                            comparison_id=f"cmp-{cid}-{_pass}-{i}",
                            creator_a=a["id"],
                            creator_b=b["id"],
                            creator_c=d["id"],
                            kind="random",
                            seed_group=a["group"],
                            expected_modality="caption_terms",
                            target_k=args.target_k,
                        )
                    )
                    n += 1
                    if n >= args.max_comparisons:
                        break
                if n >= args.max_comparisons:
                    break
            if n >= args.max_comparisons:
                break

        for b in make_boundary_comparisons(creators, rng, args.max_comparisons // 4):
            s.add(
                Comparison(
                    comparison_id=b["comparison_id"],
                    creator_a=b["creator_a"],
                    creator_b=b["creator_b"],
                    creator_c=b["creator_c"],
                    kind="boundary",
                    seed_group=b["seed_group"],
                    expected_modality="caption_terms",
                    target_k=args.target_k,
                )
            )
        for g in make_gold_items(creators, rng, max_gold=max(1, args.max_comparisons // 20)):
            s.add(
                Comparison(
                    comparison_id=g["comparison_id"],
                    creator_a=g["creator_a"],
                    creator_b=g["creator_b"],
                    creator_c=g["creator_c"],
                    kind="random",
                    seed_group=g["seed_group"],
                    status="gold",  # excluded from the scored pool; recycled
                    target_k=args.target_k,
                )
            )
            s.add(GoldItem(comparison_id=g["comparison_id"], known_odd=g["known_odd"]))
        s.commit()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="swipe_anchor.export.build_demo")
    p.add_argument(
        "--pipeline-db", default=os.environ.get("PIPELINE_DB", "data/inst2vec.db")
    )
    p.add_argument("--case", default="sandwich")
    p.add_argument("--media-base", default="/media")
    p.add_argument("--clips-per-user", type=int, default=3)
    p.add_argument("--target-k", type=int, default=5)
    p.add_argument("--max-comparisons", type=int, default=500)
    _build(p.parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
