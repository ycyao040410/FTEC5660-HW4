"""Reproduce the HW4 public-set ablation using the released data and scorer."""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from statistics import mean

import hw4


BASELINE = hw4._Options(False, False, False, False, False, False)
STEPS = [
    ("B0", "Naive RAG", None),
    ("V1", "Structure-aware chunks", "structured_chunks"),
    ("V2", "Normalized lexical tokens", "normalized_tokens"),
    ("V3", "Metadata routing", "metadata_routing"),
    ("V4", "Evidence-aware ranking", "evidence_ranking"),
    ("V5", "Comparison balancing", "company_balancing"),
    ("V6", "Evidence gate and abstention", "abstention"),
]
BEGIN = "<!-- ABLATION_RESULTS_BEGIN -->"
END = "<!-- ABLATION_RESULTS_END -->"


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate(corpus, questions, gold, options, use_generator, cache, workers,
             official_rows=None):
    started = time.perf_counter()
    hw4._OPTIONS = options
    hw4._COMPANIES.clear()
    hw4._BROKERS.clear()
    chunks = [c for source, html in corpus.items() for c in hw4.chunk_document(source, html)]
    assert chunks and all(len(c.page_content) <= hw4.MAX_CHUNK_CHARS for c in chunks)
    index = hw4.build_index(chunks)
    contexts = {qid: hw4.retrieve(index, question) for qid, question in questions.items()}
    allowed = {id(c) for c in chunks}
    assert all(len(context) <= hw4.MAX_RETRIEVED and
               all(id(c) in allowed for c in context) for context in contexts.values())
    retrieval_seconds = time.perf_counter() - started

    answers = {}
    if official_rows is not None:
        for qid, question in questions.items():
            row = official_rows[qid]
            sources = ";".join(sorted({c.metadata["source"] for c in contexts[qid]}))
            evidence = hw4.retrieval_hit(contexts[qid], gold[qid]) if gold[qid]["evidence"] else ""
            if (row["question"] != question or row["retrieved"] != sources or row["valid"] != "True"
                    or row["retrieval_hit"] != str(evidence) or not row["answer"].strip()):
                raise ValueError(f"The official results file does not match V6 at {qid}.")
            answers[qid] = row["answer"]
    elif use_generator:
        pending = {}
        for qid, question in questions.items():
            context = contexts[qid]
            cache_key = question, hw4.format_context(context)
            if not context:
                answers[qid] = "NOT_FOUND"
            elif cache_key in cache:
                answers[qid] = cache[cache_key]
            else:
                pending[qid] = cache_key
        with ThreadPoolExecutor(max_workers=workers) as pool:
            generated = pool.map(lambda qid: hw4.generate(questions[qid], contexts[qid]), pending)
            for qid, answer in zip(pending, generated):
                if not answer:
                    raise RuntimeError(f"The fixed generator returned an empty response for {qid}.")
                answers[qid] = answer
                cache[pending[qid]] = answer

    rows = []
    for qid, question in questions.items():
        expected = gold[qid]
        context = contexts[qid]
        parsed = hw4.parse_query(question)
        answer = answers.get(qid, "")
        correct = hw4.answer_correct(answer, expected) if use_generator else ""
        rows.append({
            "id": qid, "category": expected["category"], "question": question,
            "parsed": json.dumps(parsed), "parse_score": round(hw4.parse_score(parsed, expected), 3),
            "retrieved": ";".join(sorted({c.metadata["source"] for c in context})),
            "valid": True, "retrieval_hit": hw4.retrieval_hit(context, expected) if expected["evidence"] else "",
            "answer": answer, "correct": correct, "expected": expected["answer"] if correct is False else "",
        })

    answerable = [r for r in rows if gold[r["id"]]["evidence"]]
    unanswerable = [r for r in rows if not gold[r["id"]]["evidence"]]
    comparisons = [r for r in answerable if r["category"] == "comparison"]
    single_fact = [r for r in answerable if len(gold[r["id"]]["evidence"]) == 1]
    return rows, {
        "queries": len(rows),
        "chunks": len(chunks),
        "recall_at_5": mean(r["retrieval_hit"] for r in answerable),
        "first_passage_hit": mean(hw4.retrieval_hit(contexts[r["id"]][:1], gold[r["id"]])
                                  for r in single_fact) if single_fact else None,
        "comparison_first_two_hit": mean(hw4.retrieval_hit(contexts[r["id"]][:2], gold[r["id"]])
                                         for r in comparisons) if comparisons else None,
        "comparison_recall": mean(r["retrieval_hit"] for r in comparisons) if comparisons else None,
        "kept": mean(bool(r["retrieved"]) for r in answerable),
        "rejected": mean(not r["retrieved"] for r in unanswerable),
        "accuracy": mean(r["correct"] for r in rows) if use_generator else None,
        "offline_seconds": round(retrieval_seconds, 3),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }


def pct(value):
    return "—" if value is None else f"{100 * value:.1f}%"


def update_readme(path: Path, summary: list[dict], final_rows: list[dict]) -> None:
    text = path.read_text(encoding="utf-8")
    if text.count(BEGIN) != 1 or text.count(END) != 1:
        raise ValueError(f"Expected exactly one pair of ablation markers in {path}.")
    before, remainder = text.split(BEGIN, 1)
    _, after = remainder.split(END, 1)
    cols = ("| Variant | Single component changed | Recall@5 | Single-fact top-1 |"
            " Comparison top-2 | Comparison top-5 | Kept | Rejected | Answer accuracy |")
    sep = "|---|---|---:|---:|---:|---:|---:|---:|---:|"
    lines = [cols, sep]
    for row in summary:
        metrics = [pct(row[key]) for key in ("recall_at_5", "first_passage_hit", "comparison_first_two_hit",
                                             "comparison_recall", "kept", "rejected", "accuracy")]
        lines.append("| " + " | ".join([row["variant"], row["component"], *metrics]) + " |")

    if summary[-1]["accuracy"] is None:
        lines += ["", "Answer accuracy is pending a run of the fixed DeepSeek generator."]
    else:
        changes = []
        for previous, current in zip(summary, summary[1:]):
            delta = 100 * (current["accuracy"] - previous["accuracy"])
            changes.append(f"{current['variant']} {delta:+.1f} pp")
        lines += ["", "Observed answer-accuracy change relative to the preceding row — "
                  + ", ".join(changes) + "."]
        scored = [r for r in final_rows if r["correct"] is False]
        failures = [r["id"] for r in scored]
        with_evidence = [r["id"] for r in scored if r["retrieval_hit"] is True]
        examples = ", ".join(f"`{qid}`" for qid in failures) if failures else "none"
        supported = ", ".join(f"`{qid}`" for qid in with_evidence) if with_evidence else "none"
        lines += ["", f"Final variant answer errors ({len(failures)}/{len(final_rows)}) — {examples}. "
                  f"Errors with a full retrieval hit — {supported}."]
        if scored:
            lines += ["", "| Question | Returned answer | Expected answer | Evidence hit |",
                      "|---|---|---|---|"]
            def safe(value):
                return str(value).replace("|", "\\|").replace("\n", " ").strip()

            for row in scored:
                items = [row["id"], safe(row["answer"]) or "(empty)", safe(row["expected"]),
                         str(row["retrieval_hit"]) if row["retrieval_hit"] != "" else "—"]
                lines.append("| " + " | ".join(items) + " |")
    path.write_text(before + BEGIN + "\n\n" + "\n".join(lines) + "\n\n" + END + after, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-folder", type=Path, default=Path("public_test"))
    parser.add_argument("--output-dir", type=Path, default=Path("ablation_results"))
    parser.add_argument("--with-generator", action="store_true")
    parser.add_argument("--update-readme", type=Path)
    parser.add_argument("--official-results", type=Path,
                        help="Reuse this official runner CSV's answers for the final variant.")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.with_generator and not os.environ.get("DEEPSEEK_API_KEY"):
        parser.error("--with-generator requires DEEPSEEK_API_KEY.")
    if args.official_results and not args.with_generator:
        parser.error("--official-results is used with --with-generator.")
    if args.workers < 1:
        parser.error("--workers must be positive.")

    corpus = hw4.load_corpus(args.data_folder / "corpus")
    questions = hw4.load_json(args.data_folder / "questions.json")
    gold = hw4.load_json(args.data_folder / "ground_truth.json")
    if not corpus or not questions or not gold or set(questions) != set(gold):
        parser.error("Provide the released corpus, questions.json and matching ground_truth.json.")
    official_rows = None
    if args.official_results:
        with args.official_results.open(newline="", encoding="utf-8") as handle:
            official_rows = {row["id"]: row for row in csv.DictReader(handle)}
        if set(official_rows) != set(questions):
            parser.error("The official results file must cover the same question IDs.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    options = BASELINE
    cache = {}
    final_rows = []
    for variant, component, field in STEPS:
        if field:
            options = replace(options, **{field: True})
        reuse = official_rows if variant == STEPS[-1][0] else None
        rows, metrics = evaluate(corpus, questions, gold, options, args.with_generator, cache,
                                 args.workers, reuse)
        summary.append({"variant": variant, "component": component, **metrics})
        write_csv(args.output_dir / f"{variant}_results.csv", rows)
        final_rows = rows
        write_csv(args.output_dir / "summary.csv", summary)
        print(f"{variant:>2} | Recall@5 {pct(metrics['recall_at_5']):>6} | "
              f"Kept {pct(metrics['kept']):>6} | Rejected {pct(metrics['rejected']):>6} | "
              f"Accuracy {pct(metrics['accuracy']):>6}", flush=True)
    if args.update_readme:
        update_readme(args.update_readme, summary, final_rows)
    print(f"Wrote {args.output_dir / 'summary.csv'}")
    if args.with_generator:
        print(f"Generator usage: {hw4.GENERATOR_USAGE.summary()}")


if __name__ == "__main__":
    main()
