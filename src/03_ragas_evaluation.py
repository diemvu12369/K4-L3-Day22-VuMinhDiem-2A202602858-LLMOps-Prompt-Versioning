"""
Bước 3 — RAGAS Evaluation
===========================
NHIỆM VỤ:
  1. Chạy 50 QA pairs qua CẢ 2 prompt version, lưu answers + contexts
  2. Tạo EvaluationDataset với các SingleTurnSample object
  3. Đánh giá với 4 RAGAS metrics: faithfulness, answer_relevancy,
     context_recall, context_precision
  4. In bảng so sánh V1 vs V2
  5. Lưu kết quả vào data/ragas_report.json

DELIVERABLE: faithfulness ≥ 0.8 cho ít nhất 1 prompt version
             + file data/ragas_report.json được tạo ra

⏰ LƯU Ý: Bước này mất ~15-30 phút. Hãy bắt đầu sớm!

Khả năng chạy tiếp (resume): câu trả lời RAG và điểm từng sample được lưu vào
data/ragas_cache/ sau mỗi lô → nếu bị ngắt (hết quota, mất mạng) chỉ cần chạy lại,
script bỏ qua phần đã xong. Xoá thư mục đó để chạy lại từ đầu.
"""
import sys
import json
import importlib
import warnings
warnings.filterwarnings("ignore")

from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import config  # ⚠️ phải import trước LangChain

import numpy as np
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision
from ragas.run_config import RunConfig

from utils.llm_factory import get_llm, get_eval_llm, get_embeddings
from utils.data_loader import load_knowledge_base, split_text, build_vectorstore
from qa_pairs import QA_PAIRS

METRICS    = ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]
CACHE_DIR  = Path(__file__).parent.parent / "data" / "ragas_cache"
BATCH_SIZE = 10   # số sample mỗi lần gọi evaluate() — lưu cache sau mỗi lô


# ── 1. Prompt Templates (dùng chung với Bước 2) ───────────────────────────
# Import SYSTEM_V1 / SYSTEM_V2 từ Bước 2 thay vì copy tay → đảm bảo prompt được
# đánh giá giống hệt prompt đã push lên Hub (tên module bắt đầu bằng số nên
# phải dùng importlib).
_step2 = importlib.import_module("02_prompt_hub_ab_routing")
SYSTEM_V1 = _step2.SYSTEM_V1
SYSTEM_V2 = _step2.SYSTEM_V2

PROMPT_V1 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_V1),
    ("human",  "{question}"),
])

PROMPT_V2 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_V2),
    ("human",  "{question}"),
])

PROMPTS = {"v1": PROMPT_V1, "v2": PROMPT_V2}


# ── 2. Setup Vectorstore ───────────────────────────────────────────────────
def setup_vectorstore():
    """Tái sử dụng — tạo FAISS vectorstore từ knowledge base."""
    embeddings  = get_embeddings()
    text        = load_knowledge_base()
    chunks      = split_text(text)
    return build_vectorstore(chunks, embeddings)


# ── 3. Chạy RAG và thu thập kết quả ───────────────────────────────────────
def run_rag(retriever, llm, prompt, question: str) -> dict:
    """
    Chạy RAG chain cho 1 câu hỏi.

    ⚠️ contexts là LIST of strings, KHÔNG phải string đã ghép — RAGAS cần từng
    đoạn riêng để tính context_recall và context_precision.

    Trả về: {"answer": str, "contexts": list[str]}
    """
    docs = retriever.invoke(question)
    contexts = [doc.page_content for doc in docs]
    ctx_str = "\n\n".join(contexts)  # chỉ dùng để điền {context} trong prompt

    answer = (prompt | llm | StrOutputParser()).invoke({
        "context":  ctx_str,
        "question": question,
    })

    return {"answer": answer, "contexts": contexts}


def _load_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def _save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def collect_rag_outputs(vectorstore, prompt_version: str) -> list:
    """
    Chạy tất cả 50 QA pairs qua prompt version được chỉ định.
    Trả về: list of dict với keys: question, reference, answer, contexts

    Kết quả được cache ở data/ragas_cache/rag_outputs_<version>.json (lưu sau mỗi câu).
    """
    cache_path = CACHE_DIR / f"rag_outputs_{prompt_version}.json"
    results    = _load_json(cache_path, [])
    if len(results) >= len(QA_PAIRS):
        print(f"\n♻️  Dùng {len(results)} câu trả lời {prompt_version} đã cache.")
        return results[:len(QA_PAIRS)]

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm       = get_llm()
    prompt    = PROMPTS[prompt_version]

    print(f"\n🚀 Đang chạy {len(QA_PAIRS)} câu hỏi với prompt {prompt_version} ...")

    for i, qa in enumerate(QA_PAIRS, 1):
        if i <= len(results):
            continue
        out = run_rag(retriever, llm, prompt, qa["question"])

        results.append({
            "question":  qa["question"],
            "reference": qa["reference"],
            "answer":    out["answer"],
            "contexts":  out["contexts"],
        })
        _save_json(cache_path, results)
        print(f"  [{i:02d}/{len(QA_PAIRS)}] {qa['question'][:60]}")

    return results


# ── 4. Tạo RAGAS EvaluationDataset ────────────────────────────────────────
def build_ragas_dataset(rag_results: list) -> EvaluationDataset:
    """
    Chuyển đổi kết quả RAG thành RAGAS EvaluationDataset.

    Mỗi SingleTurnSample cần 4 trường:
      user_input         → câu hỏi
      response           → câu trả lời đã tạo
      retrieved_contexts → list[str] các đoạn đã retrieve
      reference          → đáp án chuẩn (ground truth)
    """
    samples = [
        SingleTurnSample(
            user_input=r["question"],
            response=r["answer"],
            retrieved_contexts=r["contexts"],
            reference=r["reference"],
        )
        for r in rag_results
    ]
    return EvaluationDataset(samples=samples)


# ── 5. Chạy RAGAS Evaluation ──────────────────────────────────────────────
def _clean(v):
    """Chuẩn hoá điểm: NaN/None → None để lưu JSON và bỏ qua khi tính trung bình."""
    return None if v is None or (isinstance(v, float) and np.isnan(v)) else float(v)


def run_ragas_eval(results_by_version: dict) -> dict:
    """
    Đánh giá kết quả RAG của MỌI version với 4 RAGAS metrics.
    Trả về: {version: {metric_name: mean_score}}

    - Sample của V1 và V2 được xếp XEN KẼ (v1#1, v2#1, v1#2, ...) → nếu giám khảo
      phải đổi model giữa chừng (hết quota), cả 2 version chịu ảnh hưởng như nhau.
    - Chạy theo lô BATCH_SIZE sample và cache điểm từng sample → resume được.
    """
    llm_eval = get_eval_llm()
    emb_eval = get_embeddings()

    # Gemini không hỗ trợ n-completions → answer_relevancy gọi LLM `strictness` lần
    # cho mỗi sample. Giảm còn 1 để vừa quota free tier (provider khác giữ mặc định 3).
    if config.PROVIDER == "gemini":
        answer_relevancy.strictness = 1

    cache_path = CACHE_DIR / "sample_scores.json"
    cache      = _load_json(cache_path, {})

    items = []
    for i in range(len(QA_PAIRS)):
        for version, results in results_by_version.items():
            items.append((f"{version}:{i}", results[i]))

    # Sample chưa có điểm, hoặc có metric bị lỗi (None) → chấm (lại)
    pending = [(k, r) for k, r in items
               if k not in cache or any(cache[k].get(m) is None for m in METRICS)]
    print(f"\n📐 RAGAS: {len(items) - len(pending)}/{len(items)} sample đã có điểm, "
          f"còn {len(pending)} sample cần chấm ...")

    for b in range(0, len(pending), BATCH_SIZE):
        batch   = pending[b:b + BATCH_SIZE]
        dataset = build_ragas_dataset([r for _, r in batch])
        judge   = getattr(llm_eval, "active_model", type(llm_eval).__name__)
        print(f"\n  Lô {b // BATCH_SIZE + 1}/{-(-len(pending) // BATCH_SIZE)} "
              f"({len(batch)} sample, giám khảo: {judge})")

        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
            llm=llm_eval,
            embeddings=emb_eval,
            # Ít worker + timeout dài: hợp với rate limit của free tier; giám khảo dự phòng
            # (vd. Gemma) có thể mất ~3-4 phút cho 1 job faithfulness.
            run_config=RunConfig(max_workers=4, timeout=600, max_retries=10, max_wait=90),
            show_progress=False,
        )

        for j, (key, _) in enumerate(batch):
            cache[key] = {m: _clean(result[m][j]) for m in METRICS}
            cache[key]["judge"] = getattr(llm_eval, "active_model", judge)
        _save_json(cache_path, cache)

    # Tổng hợp: RAGAS trả về điểm từng sample → lấy trung bình, bỏ qua None
    all_scores = {}
    for version in results_by_version:
        keys   = [f"{version}:{i}" for i in range(len(QA_PAIRS))]
        scores = {}
        for m in METRICS:
            vals = [cache[k][m] for k in keys if k in cache and cache[k].get(m) is not None]
            scores[m] = float(np.mean(vals)) if vals else 0.0
        scores["n_scored"] = {m: sum(1 for k in keys if k in cache and cache[k].get(m) is not None)
                              for m in METRICS}
        all_scores[version] = scores

        print(f"\n📊 Kết quả RAGAS — Prompt {version.upper()}:")
        for m in METRICS:
            star = " ⭐" if m == "faithfulness" and scores[m] >= 0.8 else ""
            print(f"  {m:30s}: {scores[m]:.4f}  (n={scores['n_scored'][m]}){star}")

    return all_scores


# ── 6. Main ────────────────────────────────────────────────────────────────
def _keep_awake():
    """Windows: chặn máy tự ngủ trong lúc chạy (ngủ giữa chừng làm request bị treo)."""
    if sys.platform == "win32":
        import ctypes
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)


def main():
    print("=" * 60)
    print("  Bước 3: RAGAS Evaluation")
    print("=" * 60)

    if not config.validate():
        sys.exit(1)

    _keep_awake()

    vectorstore = setup_vectorstore()

    # Thu thập kết quả RAG cho cả V1 và V2
    v1_results = collect_rag_outputs(vectorstore, "v1")
    v2_results = collect_rag_outputs(vectorstore, "v2")

    # Chạy RAGAS evaluation (V1 và V2 chấm xen kẽ)
    scores    = run_ragas_eval({"v1": v1_results, "v2": v2_results})
    v1_scores = scores["v1"]
    v2_scores = scores["v2"]

    # In bảng so sánh
    print("\n" + "=" * 65)
    print(f"  {'Metric':30s}  {'V1':>8}  {'V2':>8}  Winner")
    print("=" * 65)
    for metric in METRICS:
        s1, s2  = v1_scores[metric], v2_scores[metric]
        winner  = "= Tie" if abs(s1 - s2) < 1e-4 else ("← V1" if s1 > s2 else "← V2")
        print(f"  {metric:30s}  {s1:>8.4f}  {s2:>8.4f}  {winner}")

    # Kiểm tra mục tiêu
    best_faith = max(v1_scores["faithfulness"], v2_scores["faithfulness"])
    if best_faith >= 0.8:
        print(f"\n✅ Đạt mục tiêu: faithfulness = {best_faith:.4f} ≥ 0.8")
    else:
        print(f"\n⚠️  Chưa đạt mục tiêu ({best_faith:.4f} < 0.8).")
        print("   Gợi ý: giảm chunk_size, tăng k, hoặc điều chỉnh prompt.")

    sample_cache = _load_json(CACHE_DIR / "sample_scores.json", {})
    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "provider": config.PROVIDER,
        "generator_model": getattr(get_llm(), "model", None),
        "judge_models_used": sorted({v.get("judge") for v in sample_cache.values() if v.get("judge")}),
        "num_samples": len(QA_PAIRS),
        "prompt_v1_scores": {m: v1_scores[m] for m in METRICS},
        "prompt_v2_scores": {m: v2_scores[m] for m in METRICS},
        "n_scored": {"v1": v1_scores["n_scored"], "v2": v2_scores["n_scored"]},
        "target_met": best_faith >= 0.8,
    }
    report_path = Path(__file__).parent.parent / "data" / "ragas_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"💾 Đã lưu báo cáo vào {report_path}")


if __name__ == "__main__":
    main()
