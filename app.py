"""BNSF Document Assistant — Streamlit Databricks App (Template)."""

import os
import re
import json
import time
import uuid
import base64
import functools
from concurrent.futures import ThreadPoolExecutor

import requests

from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementState, Disposition, Format
from databricks.sdk.service.serving import ChatMessage, ChatMessageRole

# ── Required config (fail fast if .env is not set up) ────────────────────────

DATASET_NAME           = os.environ["DATASET_NAME"]
DOMAIN                 = os.environ["DOMAIN"]
SME_EVAL_TABLE         = os.environ["SME_EVAL_TABLE"]
EMBEDDING_ENDPOINT_URL = os.environ["EMBEDDING_ENDPOINT_URL"]
DOC_METADATA_TABLE     = os.environ["DOC_METADATA_TABLE"]
DOC_CHUNKS_TABLE       = os.environ["DOC_CHUNKS_TABLE"]
DOC_IMGS_TABLE         = os.environ["DOC_IMGS_TABLE"]
DOC_PAGE_IMGS_TABLE    = os.environ["DOC_PAGE_IMGS_TABLE"]

# ── Optional config ──────────────────────────────────────────────────────────

DATABRICKS_HOST  = os.environ.get("DATABRICKS_HOST", "").rstrip("/")
WAREHOUSE_ID     = os.environ.get("DATABRICKS_WAREHOUSE_ID", "")
PAGE_TITLE       = os.environ.get("PAGE_TITLE", "BNSF Document Assistant")
SIDEBAR_HEADING  = os.environ.get("SIDEBAR_HEADING", "Document Assistant")
WELCOME_TITLE    = os.environ.get("WELCOME_TITLE", "Welcome to the BNSF Document Assistant")
WELCOME_SUBTITLE = os.environ.get("WELCOME_SUBTITLE", "Ask a detailed question about your documents.")
CHAT_PLACEHOLDER = os.environ.get("CHAT_PLACEHOLDER", "Ask a question...")

# ── Derived constants ────────────────────────────────────────────────────────

DATASETS = {
    DATASET_NAME: {
        "domain": DOMAIN,
    }
}

LLM_MODELS = {
    "Claude Sonnet 4.6":  "databricks-claude-sonnet-4-6",
    "Claude Opus 4.6":    "databricks-claude-opus-4-6",
    "Claude Sonnet 4.5":  "databricks-claude-sonnet-4-5",
    "Claude Haiku 4.5":   "databricks-claude-haiku-4-5",
    "GPT-5 (Azure)":      "aoai-instances",
    "GPT OSS 120B":       "databricks-gpt-oss-120b",
    "GPT OSS 20B":        "databricks-gpt-oss-20b",
    "Llama 4 Maverick":   "databricks-llama-4-maverick",
    "Qwen3 Next 80B":     "databricks-qwen3-next-80b-a3b-instruct",
    "Gemma 3 12B":        "databricks-gemma-3-12b",
}
DEFAULT_MODEL = "Claude Sonnet 4.6"

SYNTHESIS_SYSTEM_PROMPT = (
    "You are a BNSF railroad expert assistant. "
    "Answer the user's question using the document excerpts provided below. "
    "You may reason from what the excerpts clearly imply, not just what they state word-for-word — "
    "but do not introduce facts that cannot be traced back to the excerpts.\n\n"
    "Structure your answer as follows:\n"
    "1. If the excerpts DO NOT contain a relevant answer: state that clearly at the very beginning "
    "(e.g. 'The available documents do not address this question.'), then provide any partial "
    "information from the excerpts that may still be useful.\n"
    "2. If the excerpts DO contain a relevant answer: lead with the answer, then append any scope "
    "notes or caveats at the end only if the excerpts genuinely limit applicability "
    "(e.g. the rule applies to a specific location or craft different from what was asked). "
    "Do not add scope notes for general questions where no limitation exists.\n"
    "3. If the excerpts contain rules from MULTIPLE different agreements or crafts that address "
    "the same question, present EACH one separately and clearly labeled by its agreement or craft "
    "(e.g. 'Under the [Agreement Name]:...'). Do not pick just one — show the complete picture.\n\n"
    "Respond with ONLY a JSON object in this exact format — no markdown, no code fences:\n"
    '{"answer": "your full answer here", '
    '"claims": ["first atomic factual claim.", "second atomic factual claim.", ...]}\n'
    "Each item in claims must be one distinct, self-contained factual assertion from your answer."
)

# ── Cached resources ──────────────────────────────────────────────────────────

@st.cache_resource
def _workspace_client() -> WorkspaceClient:
    return WorkspaceClient()

# ── Embedding model ───────────────────────────────────────────────────────────

def _get_embeddings(texts: list[str]) -> list[list[float]]:
    """Batch-embed multiple texts in a single API call."""
    from urllib.parse import urlparse
    path = urlparse(EMBEDDING_ENDPOINT_URL).path
    resp = _workspace_client().api_client.do(
        "POST", path,
        body={"input": texts},
    )
    return [item["embedding"] for item in resp["data"]]

def _get_embedding(text: str) -> list[float]:
    """Embed a single text."""
    return _get_embeddings([text])[0]

# ── LLM helper ────────────────────────────────────────────────────────────────

_ROLE_MAP = {
    "system":    ChatMessageRole.SYSTEM,
    "user":      ChatMessageRole.USER,
    "assistant": ChatMessageRole.ASSISTANT,
}

def _chat(model: str, messages: list[dict]) -> str:
    """Call a Databricks serving endpoint. Auth handled automatically by WorkspaceClient."""
    sdk_messages = [
        ChatMessage(role=_ROLE_MAP[m["role"]], content=m["content"])
        for m in messages
    ]
    resp = _workspace_client().serving_endpoints.query(name=model, messages=sdk_messages)
    return resp.choices[0].message.content

# ── SQL execution ─────────────────────────────────────────────────────────────

def _exec_sql(sql: str) -> list[dict]:
    wc = _workspace_client()
    resp = wc.statement_execution.execute_statement(
        warehouse_id=WAREHOUSE_ID,
        statement=sql,
        wait_timeout="50s",
        disposition=Disposition.EXTERNAL_LINKS,
        format=Format.JSON_ARRAY,
    )
    delay = 0.1
    while resp.status.state in (StatementState.PENDING, StatementState.RUNNING):
        time.sleep(min(delay, 2.0))
        delay = min(delay * 1.5, 2.0)
        resp = wc.statement_execution.get_statement(resp.statement_id)
    if resp.status.state != StatementState.SUCCEEDED:
        err = resp.status.error
        detail = err.message if err else "no error detail"
        raise RuntimeError(f"SQL failed [{resp.status.state}]: {detail}")
    if not resp.manifest or not resp.result:
        return []

    cols         = [c.name for c in resp.manifest.schema.columns]
    total_chunks = resp.manifest.total_chunk_count or 1
    rows         = []

    for chunk_index in range(total_chunks):
        if chunk_index == 0:
            chunk_result = resp.result
        else:
            chunk_result = wc.statement_execution.get_statement_result_chunk_n(
                resp.statement_id, chunk_index
            )
        for link in (chunk_result.external_links or []):
            data = requests.get(link.external_link, timeout=120).json()
            rows.extend(dict(zip(cols, row)) for row in data)

    return rows

# ── Document list (for sidebar filter) ───────────────────────────────────────

# ── Retrieval ─────────────────────────────────────────────────────────────────

@functools.lru_cache(maxsize=256)
def retrieve(query: str, top_k_docs: int, top_k_chunks: int, selected_guids: tuple = ()) -> list[dict]:
    """Two-stage retrieval:
      1. Score all chunks, pick the top_k_docs documents by their best chunk score.
      2. Return the top_k_chunks chunks from within those documents only.
    Single SQL query using a CTE so the dot-product is computed once."""
    q_vec = _get_embedding(query)

    vec_sql      = "ARRAY(" + ", ".join(str(v) for v in q_vec) + ")"
    doc_filter   = (
        "AND FILE_GUID IN (" + ", ".join(f"'{g}'" for g in selected_guids) + ")"
        if selected_guids else ""
    )
    chunks_per_doc = max(1, top_k_chunks // top_k_docs)
    top_chunks = _exec_sql(f"""
            WITH scored AS (
                SELECT FILE_GUID, CHUNK_IDX, CHUNK_TXT, CHUNK_ROLE, PAGE_NUMS,
                       aggregate(
                           zip_with(CHUNK_VCTR, {vec_sql}, (x, y) -> x * y),
                           CAST(0 AS DOUBLE), (acc, v) -> acc + v
                       ) AS dot_product
                FROM {DOC_CHUNKS_TABLE}
                WHERE CHUNK_VCTR IS NOT NULL {doc_filter}
            ),
            top_docs AS (
                SELECT FILE_GUID
                FROM scored
                GROUP BY FILE_GUID
                ORDER BY MAX(dot_product) DESC
                LIMIT {top_k_docs}
            ),
            per_doc_ranked AS (
                SELECT s.FILE_GUID, s.CHUNK_IDX, s.CHUNK_TXT, s.CHUNK_ROLE,
                       s.PAGE_NUMS, s.dot_product,
                       ROW_NUMBER() OVER (
                           PARTITION BY s.FILE_GUID
                           ORDER BY s.dot_product DESC
                       ) AS doc_rank
                FROM scored s
                INNER JOIN top_docs d ON s.FILE_GUID = d.FILE_GUID
            )
            SELECT p.FILE_GUID, p.CHUNK_IDX, p.CHUNK_TXT, p.CHUNK_ROLE,
                   p.PAGE_NUMS, p.dot_product, m.FILE_NM
            FROM per_doc_ranked p
            INNER JOIN {DOC_METADATA_TABLE} m ON p.FILE_GUID = m.FILE_GUID
            WHERE p.doc_rank <= {chunks_per_doc}
            ORDER BY p.dot_product DESC
        """)

    for r in top_chunks:
        if isinstance(r.get("PAGE_NUMS"), str):
            try:
                r["PAGE_NUMS"] = json.loads(r["PAGE_NUMS"])
            except (ValueError, TypeError):
                r["PAGE_NUMS"] = []

    # Fetch image descriptions only for the returned GUIDs + pages
    guids_lit = ", ".join(f"'{r['FILE_GUID']}'" for r in top_chunks)
    imgs = _exec_sql(f"""
        SELECT FILE_GUID, DOC_PAGE_NUM, IMG_DESC_TXT
        FROM {DOC_IMGS_TABLE}
        WHERE FILE_GUID IN ({guids_lit}) AND IMG_DESC_TXT IS NOT NULL
    """) if top_chunks else []

    img_lookup: dict[str, list[tuple]] = {}
    for img in imgs:
        img_lookup.setdefault(img["FILE_GUID"], []).append(
            (img["DOC_PAGE_NUM"], img["IMG_DESC_TXT"])
        )

    results = []
    for chunk in top_chunks:
        page_nums = chunk.get("PAGE_NUMS") or []
        fig_descs = "\n".join(
            desc for page, desc in img_lookup.get(chunk["FILE_GUID"], [])
            if page in page_nums
        )
        results.append({
            "source_key": f"{chunk['FILE_GUID']}|{chunk['CHUNK_IDX']}",
            "chunk_txt":  chunk["CHUNK_TXT"],
            "chunk_role": chunk["CHUNK_ROLE"],
            "fig_descs":  fig_descs,
            "file_guid":  chunk["FILE_GUID"],
            "file_nm":    chunk.get("FILE_NM", ""),
            "page_nums":  page_nums,
        })

    return results

def cite(chunks: list[dict]) -> list[dict]:
    """Build citation rows directly from retrieved chunks.
    Queries DOC_METADATA_TABLE for file names and DOC_PAGE_IMGS_TABLE
    for page image URLs — no SQL function dependency."""
    if not chunks:
        return []

    # Aggregate page numbers per FILE_GUID — normalise to int to avoid
    # type mismatches between JSON-parsed chunks and SQL-returned DOC_PAGE_NUM
    guid_pages: dict[str, set] = {}
    for c in chunks:
        guid = c["file_guid"]
        for p in (c.get("page_nums") or []):
            guid_pages.setdefault(guid, set()).add(int(p))

    guids_lit = ", ".join(f"'{g}'" for g in guid_pages)

    # File names from metadata table
    meta_rows = _exec_sql(f"""
        SELECT FILE_GUID, FILE_NM
        FROM {DOC_METADATA_TABLE}
        WHERE FILE_GUID IN ({guids_lit})
    """)
    file_names = {r["FILE_GUID"]: r["FILE_NM"] for r in meta_rows}

    # Page image URLs — only for pages we actually reference
    img_rows = _exec_sql(f"""
        SELECT FILE_GUID, DOC_PAGE_NUM, BLOB_URL
        FROM {DOC_PAGE_IMGS_TABLE}
        WHERE FILE_GUID IN ({guids_lit})
    """)
    img_lookup: dict[str, dict] = {}
    for r in img_rows:
        img_lookup.setdefault(r["FILE_GUID"], {})[int(r["DOC_PAGE_NUM"])] = r["BLOB_URL"]

    results = []
    for guid, pages in guid_pages.items():
        sorted_pages = sorted(pages)
        results.append({
            "file_nm":       file_names.get(guid, guid),
            "page_nums":     sorted_pages,
            "page_img_urls": [
                img_lookup.get(guid, {}).get(p)
                for p in sorted_pages
                if img_lookup.get(guid, {}).get(p)
            ],
        })

    return results

# ── Query enhancement ─────────────────────────────────────────────────────────

def enhance_simple(query: str, domain: str, model: str) -> str:
    """Single LLM-enhanced query for standard RAG."""
    return _chat(model, [
        {
            "role": "system",
            "content": (
                f"Rewrite the user's query to maximize cosine similarity against {domain} "
                "document text chunks. Use precise railroad industry terminology. "
                "Return only the reformulated query, nothing else."
            ),
        },
        {"role": "user", "content": query},
    ])

def enhance_rrf(query: str, domain: str, model: str) -> list[str]:
    """Generate 5 diverse enhanced queries for RRF retrieval."""
    content = _chat(model, [
        {
            "role": "system",
            "content": (
                f"Generate 5 distinct reformulations of the user's query using different "
                f"railroad industry terminology, phrasing, and perspectives. Each should "
                f"target the same underlying information need but use vocabulary likely to "
                f"appear in {domain} documents. Vary the phrasing significantly across the "
                f"5 queries to maximize vocabulary coverage. "
                f'Respond with ONLY a JSON object: {{"queries": ["...", "...", "...", "...", "..."]}}'
            ),
        },
        {"role": "user", "content": query},
    ])
    start = content.find('{')
    if start != -1:
        obj, _ = json.JSONDecoder().raw_decode(content, start)
        return obj["queries"]
    return json.loads(content)["queries"]

# ── RRF re-ranking ────────────────────────────────────────────────────────────

def rrf_rerank(result_lists: list[list[dict]], k: int = 60) -> list[dict]:
    """Reciprocal Rank Fusion: score each chunk by its rank position across all result lists."""
    scores: dict[str, float] = {}
    rows:   dict[str, dict]  = {}
    for result_list in result_lists:
        for rank, row in enumerate(result_list):
            sk = row["source_key"]
            scores[sk] = scores.get(sk, 0.0) + 1.0 / (k + rank + 1)
            rows[sk]   = row
    return [rows[sk] for sk in sorted(scores, key=scores.__getitem__, reverse=True)]

# ── Synthesis ─────────────────────────────────────────────────────────────────

def _context_block(chunks: list[dict]) -> str:
    parts = []
    for c in chunks:
        label = c.get("file_nm") or c["source_key"]
        block = f"[{label}]\n{c['chunk_txt']}"
        if c.get("fig_descs"):
            block += f"\n[Figures on this page: {c['fig_descs']}]"
        parts.append(block)
    return "\n\n---\n\n".join(parts)

def synthesize(query: str, chunks: list[dict], dialog: list[dict], model: str) -> tuple[str, list[str]]:
    """Returns (answer, claims) where claims are atomic factual assertions for metric scoring."""
    context = _context_block(chunks)
    raw = _chat(model, [
        {"role": "system", "content": f"{SYNTHESIS_SYSTEM_PROMPT}\n\n--- DOCUMENT EXCERPTS ---\n{context}"},
        *[{"role": m["role"], "content": m["content"]} for m in dialog],
        {"role": "user", "content": query},
    ])
    match = re.search(r'\{.*\}', raw, re.DOTALL)
    try:
        parsed = json.loads(match.group() if match else raw)
        answer = parsed.get("answer", raw)
        claims = [c for c in parsed.get("claims", []) if isinstance(c, str) and c.strip()]
    except (ValueError, KeyError):
        answer = raw
        claims = [s.strip() for s in re.split(r'(?<=[.!?])\s+', raw.strip()) if len(s.strip()) > 15]
    return answer, claims

# ── Faithfulness metric ───────────────────────────────────────────────────────

def _sliding_window_match(sentence: str, context: str, window: int = 5) -> bool:
    words = sentence.lower().split()
    context_lower = context.lower()
    if len(words) >= window:
        for start in range(len(words) - window + 1):
            phrase = " ".join(words[start : start + window])
            if phrase in context_lower:
                return True
        return False
    return sentence.lower() in context_lower

def _jaccard(text_a: str, text_b: str) -> float:
    a = set(text_a.lower().split())
    b = set(text_b.lower().split())
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)

def _dot(a: list[float], b: list[float]) -> float:
    """Dot product == cosine similarity for unit-normalised OpenAI embeddings."""
    return sum(x * y for x, y in zip(a, b))

def compute_faithfulness(claims: list[str], chunks: list[dict]) -> dict:
    """Score each LLM claim against retrieved chunks using window match,
    Jaccard overlap, and semantic (embedding) similarity."""
    if not claims:
        return {"score": 1.0, "supported": 0, "total": 0, "claims": []}

    chunk_texts  = [c["chunk_txt"] for c in chunks]
    full_context = " ".join(chunk_texts)

    # Batch-embed all claims + chunk texts in one API call
    all_vecs   = _get_embeddings(claims + chunk_texts)
    claim_vecs = all_vecs[:len(claims)]
    chunk_vecs = all_vecs[len(claims):]

    result_claims = []
    for claim, c_vec in zip(claims, claim_vecs):
        window_hit   = _sliding_window_match(claim, full_context)
        max_jaccard  = max(_jaccard(claim, ct) for ct in chunk_texts)
        max_semantic = max(_dot(c_vec, cv) for cv in chunk_vecs)
        composite    = (
            0.4 * (1.0 if window_hit else max_jaccard) +
            0.3 * max_jaccard +
            0.3 * max_semantic
        )
        result_claims.append({
            "claim":        claim,
            "supported":    composite >= 0.60,
            "score":        round(composite, 3),
            "window_match": window_hit,
            "semantic_sim": round(max_semantic, 3),
        })

    supported = sum(1 for c in result_claims if c["supported"])
    return {
        "score":     round(supported / len(result_claims), 3),
        "supported": supported,
        "total":     len(result_claims),
        "claims":    result_claims,
    }

# ── Hallucination metric ──────────────────────────────────────────────────────

def compute_hallucination(answer: str, chunks: list[dict], faithfulness: dict) -> dict:
    """
    Hallucination detection for labor relations RAG.
    Builds on faithfulness and adds two domain-specific checks:
      1. Ungrounded sentences  — sentences not traceable to any chunk
      2. Numerical hallucination — numbers in the answer not found in any chunk
      3. Reference hallucination — Article/Rule/Section citations not in any chunk
    """
    full_context = " ".join(c["chunk_txt"] for c in chunks).lower()

    # 1. Ungrounded sentences (directly from faithfulness claims)
    ungrounded = [c for c in faithfulness.get("claims", []) if not c["supported"]]

    # 2. Numerical hallucination — critical for labor relations (rates, days, amounts)
    numbers_in_answer = set(re.findall(r'\b\d+\.?\d*\b', answer))
    hallucinated_numbers = [n for n in numbers_in_answer if n not in full_context]

    # 3. Reference hallucination — Article/Rule/Section citations not in chunks
    refs_in_answer = re.findall(
        r'(?:Article|Rule|Section|Schedule)\s+\d+', answer, re.IGNORECASE
    )
    hallucinated_refs = [ref for ref in refs_in_answer if ref.lower() not in full_context]

    hallucination_rate = round(1.0 - faithfulness.get("score", 1.0), 3)

    return {
        "hallucination_rate":      hallucination_rate,
        "ungrounded_sentences":    ungrounded,
        "hallucinated_numbers":    hallucinated_numbers,
        "hallucinated_references": hallucinated_refs,
        "has_hallucination":       bool(ungrounded or hallucinated_numbers or hallucinated_refs),
    }

# ── Full pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(
    query:        str,
    dataset_cfg:  dict,
    top_k_docs:   int,
    top_k_chunks: int,
    use_rrf:        bool,
    dialog:         list[dict],
    llm_model:      str,
    selected_guids: tuple = (),
) -> tuple[str, list[dict], list[str], list[dict]]:
    domain = dataset_cfg["domain"]

    # Step 1 — enhance query (cache in session state so same question
    # always uses the same enhanced query within a session)
    enhance_key = f"enhance_{hash((query, use_rrf, llm_model))}"
    if enhance_key not in st.session_state:
        if use_rrf:
            with st.spinner("Generating 5 query expansions…"):
                st.session_state[enhance_key] = enhance_rrf(query, domain, llm_model)
        else:
            with st.spinner("Enhancing query…"):
                st.session_state[enhance_key] = [enhance_simple(query, domain, llm_model)]
    queries = st.session_state[enhance_key]

    if use_rrf:
        with st.expander("◆ 5 query expansions", expanded=False):
            for i, q in enumerate(queries, 1):
                st.markdown(f"**{i}.** {q}")
    else:
        with st.expander("◇ Enhanced query", expanded=False):
            st.markdown(queries[0])

    # Step 2 — retrieve
    if use_rrf:
        with st.spinner(f"Retrieving across {len(queries)} query variants…"):
            with ThreadPoolExecutor(max_workers=5) as pool:
                futs = [
                    pool.submit(retrieve, q, top_k_docs, top_k_chunks, selected_guids)
                    for q in queries
                ]
                result_lists = [f.result() for f in futs]
            chunks = rrf_rerank(result_lists)[:top_k_chunks]
    else:
        with st.spinner("Retrieving…"):
            chunks = retrieve(queries[0], top_k_docs, top_k_chunks, selected_guids)

    # Step 2.5 — show retrieved chunks
    with st.expander(f"◈ {len(chunks)} retrieved chunks", expanded=False):
        for i, c in enumerate(chunks, 1):
            st.markdown(f"**{i}. `{c.get('chunk_role', 'body')}`**")
            st.markdown(c["chunk_txt"])
            if c.get("fig_descs"):
                st.caption(f"Figures: {c['fig_descs']}")
            if i < len(chunks):
                st.divider()

    # Step 3 — synthesize
    with st.spinner("Generating response…"):
        raw_response, claims = synthesize(query, chunks, dialog, llm_model)

    # Step 4 — cite deterministically from all retrieved chunks
    cite_rows = cite(chunks)

    # Step 3.5 — faithfulness and hallucination metrics (after answer is ready)
    faithfulness  = compute_faithfulness(claims, chunks)
    hallucination = compute_hallucination(raw_response.strip(), chunks, faithfulness)

    return raw_response.strip(), cite_rows, queries, chunks, faithfulness, hallucination

# ── Enhancement rendering ─────────────────────────────────────────────────────

def render_enhancement(queries: list[str], mode: str) -> None:
    if not queries:
        return
    label = "◆ 5 query expansions" if mode == "rrf" else "◇ Enhanced query"
    with st.expander(label):
        if mode == "rrf":
            for i, q in enumerate(queries, 1):
                st.markdown(f"**{i}.** {q}")
        else:
            st.markdown(queries[0])

# ── Chunk rendering ───────────────────────────────────────────────────────────

def render_chunks(chunks: list[dict]) -> None:
    if not chunks:
        return
    with st.expander(f"◈ {len(chunks)} retrieved chunks"):
        for i, c in enumerate(chunks, 1):
            st.markdown(f"**{i}. `{c.get('chunk_role', 'body')}`**")
            st.markdown(c["chunk_txt"])
            if c.get("fig_descs"):
                st.caption(f"Figures: {c['fig_descs']}")
            if i < len(chunks):
                st.divider()

# ── Citation rendering ────────────────────────────────────────────────────────

def _fetch_image_b64(url: str) -> str | None:
    """Fetch an image from blob storage via the SQL warehouse
    and return it as a base64 string Streamlit can render."""
    try:
        safe_url = url.replace("'", "''")
        rows = _exec_sql(
            f"SELECT base64(content) AS b64 FROM read_files('{safe_url}', format => 'binaryFile') LIMIT 1"
        )
        if rows and rows[0].get("b64"):
            return rows[0]["b64"]
        return None
    except Exception as e:
        st.warning(f"Could not load image: {e}")
        return None

def render_citations(cite_rows: list[dict]) -> None:
    if not cite_rows:
        return
    with st.expander(f"📄 Sources ({len(cite_rows)})"):
        for i, row in enumerate(cite_rows):
            page_nums = row.get("page_nums") or []
            img_urls  = row.get("page_img_urls") or []
            pages_str = ", ".join(str(p) for p in page_nums)

            st.markdown(
                f"**{i + 1}. {row['file_nm']}**"
                + (f" — p. {pages_str}" if pages_str else "")
            )

            with ThreadPoolExecutor() as pool:
                fetched = list(pool.map(_fetch_image_b64, img_urls))
            for j, img_b64 in enumerate(fetched):
                page_label = f"Page {page_nums[j]}" if j < len(page_nums) else f"Page {j + 1}"
                if img_b64 is None:
                    st.caption(f"{page_label} — could not load image")
                    continue
                st.image(base64.b64decode(img_b64), caption=page_label)

            if i < len(cite_rows) - 1:
                st.divider()

# ── Faithfulness rendering ────────────────────────────────────────────────────

def render_faithfulness(faithfulness: dict) -> None:
    if not faithfulness or faithfulness.get("total", 0) == 0:
        return
    score = faithfulness["score"]
    sup   = faithfulness["supported"]
    total = faithfulness["total"]
    icon  = "🟢" if score >= 0.8 else "🟡" if score >= 0.5 else "🔴"
    with st.expander(f"{icon} Faithfulness: {score:.0%}  ({sup}/{total} sentences grounded)"):
        st.progress(score)
        for c in faithfulness.get("claims", []):
            check = "✅" if c["supported"] else "❌"
            st.markdown(f"{check} {c['claim']}")
            st.caption(
                f"score: {c['score']}  · semantic: {c.get('semantic_sim', '—')}"
                + ("  · exact phrase match" if c["window_match"] else "")
            )

# ── Hallucination rendering ───────────────────────────────────────────────────

def render_hallucination(hallucination: dict) -> None:
    if not hallucination:
        return
    rate = hallucination["hallucination_rate"]
    icon = "🔴" if rate >= 0.5 else "🟡" if rate > 0 else "🟢"
    with st.expander(f"{icon} Hallucination Risk: {rate:.0%}"):
        if not hallucination["has_hallucination"]:
            st.success("No hallucinations detected.")
            return
        if hallucination["ungrounded_sentences"]:
            st.markdown("**Ungrounded sentences:**")
            for s in hallucination["ungrounded_sentences"]:
                st.markdown(f"❌ {s['claim']}")
                st.caption(f"grounding score: {s['score']}")
        if hallucination["hallucinated_numbers"]:
            st.markdown("**Numbers not found in retrieved chunks:**")
            st.caption("These figures may be invented — not in any source document.")
            for n in hallucination["hallucinated_numbers"]:
                st.markdown(f"⚠️ `{n}`")
        if hallucination["hallucinated_references"]:
            st.markdown("**Article / Rule references not in retrieved chunks:**")
            st.caption("These citations were not in the documents retrieved for this query.")
            for ref in hallucination["hallucinated_references"]:
                st.markdown(f"⚠️ `{ref}`")

# ── SME evaluation persistence ───────────────────────────────────────────────

def save_evaluation(
    response_id: str, session_id: str, quality_rating: int,
    is_relevant: bool, is_accurate: bool, notes: str,
    user_query: str, model_name: str, retrieval_mode: str,
) -> bool:
    """Upsert an SME evaluation to the project's evaluation table."""
    notes_esc = notes.replace("'", "''")
    query_esc = user_query.replace("'", "''")
    sql = f"""
    MERGE INTO {SME_EVAL_TABLE} AS t
    USING (SELECT '{response_id}' AS RESPONSE_ID) AS s
    ON t.RESPONSE_ID = s.RESPONSE_ID
    WHEN MATCHED THEN UPDATE SET
        QUALITY_RATING  = {quality_rating},
        IS_RELEVANT     = {str(is_relevant).lower()},
        IS_ACCURATE     = {str(is_accurate).lower()},
        NOTES           = '{notes_esc}',
        AZURE_LOAD_DTTM = current_timestamp()
    WHEN NOT MATCHED THEN INSERT (
        RESPONSE_ID, SESSION_ID, QUALITY_RATING, IS_RELEVANT,
        IS_ACCURATE, NOTES, USER_QUERY, MODEL_NM,
        RETRIEVAL_MODE, CRET_DT, AZURE_LOAD_DTTM
    ) VALUES (
        '{response_id}', '{session_id}', {quality_rating},
        {str(is_relevant).lower()}, {str(is_accurate).lower()},
        '{notes_esc}', '{query_esc}', '{model_name}',
        '{retrieval_mode}', cast(current_date() as date), current_timestamp()
    )"""
    try:
        _exec_sql(sql)
        return True
    except Exception as e:
        raise RuntimeError(f"save_evaluation failed: {e}") from e


def load_evaluation(response_id: str) -> dict | None:
    """Load a saved evaluation for a response, or None if not yet evaluated."""
    rows = _exec_sql(
        f"SELECT * FROM {SME_EVAL_TABLE} "
        f"WHERE RESPONSE_ID = '{response_id}'"
    )
    return rows[0] if rows else None


def render_evaluation(msg: dict) -> None:
    """Render SME evaluation form inside a collapsible expander."""
    if msg.get("role") != "assistant" or "response_id" not in msg:
        return
    rid = msg["response_id"]
    cache_key = f"eval_data_{rid}"
    saved_key  = f"eval_saved_{rid}"

    # Load from DB on first render, cache result in session state
    if cache_key not in st.session_state:
        try:
            st.session_state[cache_key] = load_evaluation(rid)
        except Exception:
            st.session_state[cache_key] = None
    existing = st.session_state[cache_key]

    label = "✎ SME Evaluation" + ("  ✓" if st.session_state.get(saved_key) else "")
    with st.expander(label):
        with st.form(key=f"eval_form_{rid}"):
            quality = st.select_slider(
                "Overall Quality",
                options=[1, 2, 3, 4, 5],
                value=int(existing["QUALITY_RATING"]) if existing and existing.get("QUALITY_RATING") else 3,
                format_func=lambda x: {
                    1: "1 — Poor", 2: "2 — Below Average", 3: "3 — Average",
                    4: "4 — Good", 5: "5 — Excellent",
                }[x],
                key=f"eval_quality_{rid}",
            )
            col1, col2 = st.columns(2)
            with col1:
                rel_val = existing and existing.get("IS_RELEVANT") in ("true", True)
                relevant = st.radio(
                    "Relevant?", ["Yes", "No"],
                    index=0 if rel_val else 1,
                    horizontal=True, key=f"eval_relevant_{rid}",
                )
            with col2:
                acc_val = existing and existing.get("IS_ACCURATE") in ("true", True)
                accurate = st.radio(
                    "Accurate?", ["Yes", "No"],
                    index=0 if acc_val else 1,
                    horizontal=True, key=f"eval_accurate_{rid}",
                )
            notes = st.text_area(
                "Notes",
                value=existing["NOTES"] if existing and existing.get("NOTES") else "",
                placeholder="Optional SME comments…",
                key=f"eval_notes_{rid}",
            )
            if st.form_submit_button("Save Evaluation"):
                try:
                    save_evaluation(
                        response_id=rid,
                        session_id=st.session_state.session_id,
                        quality_rating=quality,
                        is_relevant=(relevant == "Yes"),
                        is_accurate=(accurate == "Yes"),
                        notes=notes,
                        user_query=msg.get("user_query", ""),
                        model_name=msg.get("model_name", ""),
                        retrieval_mode=msg.get("mode", ""),
                    )
                    st.session_state[saved_key] = True
                    st.session_state[cache_key] = None  # clear cache to reload on next render
                    st.success("Evaluation saved.")
                except Exception as e:
                    st.error(f"Save failed: {e}")


# ── App ───────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title=PAGE_TITLE,
        page_icon="https://bnsf.com/favicon.ico?v=1.0b",
        layout="wide",
    )

    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', 'Segoe UI', sans-serif;
    }

    /* ── Main background ── */
    .stApp {
        background-color: #F5F6FA;
    }

    /* ── Sidebar background ── */
    [data-testid="stSidebar"] {
        background-color: #1C1C2E !important;
    }

    /* ── Logo: styled white card so the opaque PNG looks intentional ── */
    [data-testid="stSidebar"] img {
        background: #FFFFFF;
        border-radius: 10px;
        padding: 10px 14px;
        margin-bottom: 6px;
        display: block;
    }

    /* ── Sidebar base text — explicit selectors beat Streamlit's component styles ── */
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] div,
    [data-testid="stSidebar"] li {
        color: #E8EAF6 !important;
    }

    /* ── Document Assistant heading ── */
    [data-testid="stSidebar"] h3 {
        color: #FFFFFF !important;
        font-size: 1.1rem;
        font-weight: 600;
        letter-spacing: 0.02em;
        border-bottom: 2px solid #F15A22;
        padding-bottom: 6px;
        margin-bottom: 4px;
    }

    /* ── Dividers ── */
    [data-testid="stSidebar"] hr {
        border-color: rgba(255,255,255,0.2) !important;
        margin: 12px 0 !important;
    }

    /* ── Widget section labels (CORPUS, MODEL, etc.) ── */
    [data-testid="stSidebar"] label {
        color: #9BA5C2 !important;
        font-size: 0.75rem !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.07em !important;
    }

    /* ── Radio option text (undo uppercase from label rule above) ── */
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] label {
        text-transform: none !important;
        letter-spacing: normal !important;
        font-size: 0.92rem !important;
        color: #E8EAF6 !important;
        font-weight: 400 !important;
    }

    /* ── Selectbox ── */
    [data-testid="stSidebar"] [data-baseweb="select"] {
        background-color: #252542 !important;
        border: 1px solid rgba(255,255,255,0.2) !important;
        border-radius: 8px !important;
    }
    [data-testid="stSidebar"] [data-baseweb="select"] *:not(svg):not(path) {
        color: #F0F1FF !important;
        background-color: transparent !important;
    }

    /* ── Toggle label (Enhanced Retrieval text) ── */
    [data-testid="stSidebar"] .stToggle p,
    [data-testid="stSidebar"] .stToggle label,
    [data-testid="stSidebar"] [data-testid="stToggle"] p {
        color: #E8EAF6 !important;
        text-transform: none !important;
        font-size: 0.92rem !important;
        letter-spacing: normal !important;
        font-weight: 400 !important;
    }

    /* ── Settings expander ── */
    [data-testid="stSidebar"] details > summary {
        color: #C8CCE0 !important;
        font-size: 0.9rem !important;
    }
    [data-testid="stSidebar"] details > summary:hover {
        color: #FFFFFF !important;
    }
    [data-testid="stSidebar"] details > summary svg {
        fill: #C8CCE0 !important;
    }

    /* ── Sliders ── */
    [data-testid="stSidebar"] [data-testid="stSlider"] label,
    [data-testid="stSidebar"] [data-testid="stSlider"] p,
    [data-testid="stSidebar"] [data-testid="stSlider"] span {
        color: #E8EAF6 !important;
        text-transform: none !important;
        font-size: 0.9rem !important;
        letter-spacing: normal !important;
    }

    /* ── Clear conversation button → BNSF orange ── */
    [data-testid="stSidebar"] .stButton > button {
        background-color: #F15A22 !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        letter-spacing: 0.02em !important;
        transition: background-color 0.2s ease !important;
        margin-top: 4px !important;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        background-color: #D44E1A !important;
    }

    /* ── Chat messages ── */
    [data-testid="stChatMessage"] {
        border-radius: 12px !important;
        margin-bottom: 10px !important;
        padding: 14px 18px !important;
    }
    [data-testid="stChatMessage"][data-role="user"] {
        background-color: #FFF3EE !important;
        border-left: 3px solid #F15A22 !important;
    }
    [data-testid="stChatMessage"][data-role="assistant"] {
        background-color: #FFFFFF !important;
        box-shadow: 0 1px 6px rgba(0,0,0,0.07) !important;
    }

    /* ── Chat input bar ── */
    [data-testid="stChatInput"] textarea {
        border-radius: 12px !important;
        border-color: #E0E0EC !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06) !important;
        background-color: #FFFFFF !important;
    }
    [data-testid="stChatInput"] textarea:focus {
        border-color: #F15A22 !important;
        box-shadow: 0 0 0 2px rgba(241,90,34,0.2) !important;
    }

    /* ── Main area expanders ── */
    [data-testid="stExpander"] summary {
        font-weight: 500 !important;
        color: #555 !important;
    }

    /* ── Welcome state ── */
    .welcome {
        text-align: center;
        margin-top: 15vh;
        color: #999;
    }
    .welcome h2 {
        color: #444;
        font-weight: 600;
        font-size: 1.5rem;
        margin-bottom: 8px;
    }
    .welcome p {
        font-size: 1rem;
        color: #888;
    }
    </style>
    """, unsafe_allow_html=True)

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.image("https://bnsf.com/bnsf-resources/images/bnsf-logo.png", width=180)
        st.markdown(f"### {SIDEBAR_HEADING}")
        st.divider()

        dataset_name = st.radio("Corpus", list(DATASETS.keys()))
        dataset_cfg  = DATASETS[dataset_name]

        st.divider()

        model_name = st.selectbox(
            "Model",
            list(LLM_MODELS.keys()),
            index=list(LLM_MODELS.keys()).index(DEFAULT_MODEL),
        )
        llm_model = LLM_MODELS[model_name]

        st.divider()

        use_rrf = st.toggle(
            "Enhanced Retrieval (RRF)",
            help=(
                "ON — 5 query expansions generated via structured output, "
                "retrieved in parallel, re-ranked with Reciprocal Rank Fusion.\n\n"
                "OFF — Single enhanced query, standard retrieval."
            ),
        )

        with st.expander("Settings"):
            top_k_docs   = st.slider("Documents to search", min_value=1, max_value=10, value=5)
            top_k_chunks = st.slider("Chunks to return",    min_value=1, max_value=25, value=15)

        st.divider()
        if st.button("Clear conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    # ── Chat history ──────────────────────────────────────────────────────────
    if not st.session_state.messages:
        st.markdown(f"""
        <div class="welcome">
            <h2>{WELCOME_TITLE}</h2>
            <p>{WELCOME_SUBTITLE}</p>
        </div>
        """, unsafe_allow_html=True)

    # Lazy-assign response_id to any assistant messages that predate this feature
    for msg in st.session_state.messages:
        if msg["role"] == "assistant" and "response_id" not in msg:
            msg["response_id"] = str(uuid.uuid4())

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                badge = "◆ Enhanced (RRF)" if msg.get("mode") == "rrf" else "◇ Standard"
                st.caption(f"{badge} · {msg.get('model_name', '')}")
                render_enhancement(msg.get("queries", []), msg.get("mode", "standard"))
                render_chunks(msg.get("chunks", []))
                render_citations(msg.get("references", []))
                # render_faithfulness(msg.get("faithfulness", {}))
                # render_hallucination(msg.get("hallucination", {}))
                render_evaluation(msg)

    # ── Input ─────────────────────────────────────────────────────────────────
    if user_input := st.chat_input(CHAT_PLACEHOLDER):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        dialog_history = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages[:-1]
        ]

        with st.chat_message("assistant"):
            try:
                response, cite_rows, queries, chunks, faithfulness, hallucination = run_pipeline(
                    query=user_input,
                    dataset_cfg=dataset_cfg,
                    top_k_docs=top_k_docs,
                    top_k_chunks=top_k_chunks,
                    use_rrf=use_rrf,
                    dialog=dialog_history,
                    llm_model=llm_model,
                )
                st.markdown(response)
                badge = "◆ Enhanced (RRF)" if use_rrf else "◇ Standard"
                st.caption(f"{badge} · {model_name}")
                render_citations(cite_rows)
                # render_faithfulness(faithfulness)
                # render_hallucination(hallucination)
                assistant_msg = {
                    "role":         "assistant",
                    "content":      response,
                    "mode":         "rrf" if use_rrf else "standard",
                    "model_name":   model_name,
                    "queries":      queries,
                    "chunks":       chunks,
                    "references":   cite_rows,
                    "response_id":  str(uuid.uuid4()),
                    "user_query":   user_input,
                    "dataset_name": dataset_name,
                    "faithfulness": faithfulness,
                    "hallucination": hallucination,
                }
                st.session_state.messages.append(assistant_msg)
                render_evaluation(assistant_msg)
            except Exception as e:
                st.error(f"Pipeline error: {e}")
                raise

if __name__ == "__main__":
    main()
