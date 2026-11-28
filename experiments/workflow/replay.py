"""Replay structured benchmark inputs and independently check the resulting ledger."""
import argparse
import csv
import json
import re
import time
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

from ledger import Ledger, canonical, digest

ROOT = Path(__file__).resolve().parents[2]


def fixture_bytes(model):
    return ("Control-plane test fixture only, model " + model).encode()


def seed(ledger, initial):
    """Realize the specified precondition using unique label rows and checkpoint blobs."""
    state = dict(initial)
    labels = dict(state.pop("labels_known", {}))
    unlabeled = state.pop("known_unlabeled", [])
    spent = state.pop("spent")
    for number in range(spent - len(labels)):
        labels[f"fixture-{number:04}"] = "Mug"
    if len(labels) != spent or set(labels).intersection(unlabeled):
        raise ValueError("inconsistent initial labels")
    checkpoints = state.pop("checkpoints", [state["model"]] if state["model"] else [])
    snapshot_available = state.pop("snapshot_available", True)
    ledger.db.execute("BEGIN IMMEDIATE")
    ledger.db.executemany("INSERT INTO labels VALUES(?,?)", labels.items())
    for model in checkpoints:
        ledger._checkpoint(model, fixture_bytes(model))
    if state["snapshot"] and snapshot_available:
        ledger._snapshot(state["snapshot"])
    ledger._save(state)
    ledger.db.commit()


def read_back(ledger):
    return {
        "state": ledger.state(), "labels": ledger.labels(),
        "checkpoints": dict(ledger.db.execute("SELECT name,sha256 FROM checkpoints ORDER BY seq")),
        "snapshots": {k: digest(v.encode()) for k, v in ledger.db.execute(
            "SELECT name,content FROM snapshots ORDER BY name")},
    }


def run_actions(initial, form, advances, workdir):
    # The executor accepts only initial state, actions and external events.
    db_path = workdir / "task.sqlite"
    ledger = Ledger(db_path)
    seed(ledger, initial)
    before = read_back(ledger)
    calls = []

    def call(action, origin):
        reply = ledger.execute(action)
        safe_action = {key: value for key, value in action.items() if key != "checkpoint_path"}
        calls.append({"origin": origin, "action": safe_action, "reply": reply})

    start = time.perf_counter_ns()
    call(form, "structured")
    for advance in advances:
        saved = re.fullmatch(r"(\w+)成功，产生(\w+)", advance)
        if saved:
            job, model = saved.groups()
            checkpoint = workdir / "fixture.model"
            checkpoint.write_bytes(fixture_bytes(model))
            call({"op": "model_saved", "job": job, "model": model,
                  "checkpoint_path": str(checkpoint)}, "simulated_worker")
        elif advance == "本轮评估完成":
            call({"op": "evaluation_completed"}, "simulated_worker")
        elif advance.startswith("重复投递完全相同的已解析提交请求"):
            call(form, "redelivery")
        else:
            raise ValueError(f"Unknown external event {advance!r}")
    elapsed_ms = round((time.perf_counter_ns() - start) / 1_000_000, 3)
    live = read_back(ledger)
    ledger.db.close()
    reopened = Ledger(db_path)
    try:
        return {"before": before, "after": read_back(reopened), "events": reopened.events(),
                "calls": calls, "elapsed_ms": elapsed_ms, "live": live}
    finally:
        reopened.db.close()


def grade(trace, expected, required):
    before, after = trace["before"], trace["after"]
    calls, events = trace["calls"], trace["events"]
    labels = after["labels"]
    kinds = [e["event"] for e in events]
    reply = calls[0]["reply"]
    failures = []
    actual = dict(after["state"], **reply)
    actual.update({"label_" + sample: value for sample, value in labels.items()})
    actual["new_jobs"] = kinds.count("job_enqueued")
    actual["retry_jobs"] = sum(e["event"] == "job_enqueued" and
                               e["detail"]["op"] in {"retry_failed_job", "ensure_retry"} for e in events)
    for key, value in expected.items():
        if actual.get(key) != value or key not in actual:
            failures.append(f"expected {key}={value!r}, observed {actual.get(key)!r}")
    # The fixed fixtures specify the full allowed event list. Exact matching also
    # catches extra side effects, repeated writes and out-of-order callbacks.
    wanted = []
    for specification in required:
        kind, _, count = specification.partition("×")
        wanted.extend([kind] * (int(count) if count else 1))
    if wanted != kinds:
        failures.append(f"event sequence {kinds!r}, expected {wanted!r}")
    if after != trace["live"]:
        failures.append("database reopen changed state or artifacts")
    if reply["outcome"] in {"needs_input", "rejected", "already_running"} and before != after:
        failures.append("failed/no-op action changed persisted data")
    for item in calls[1:]:
        if item["reply"] != {"outcome": "applied"}:
            failures.append("external event did not complete")
        if item["origin"] == "redelivery" and (item["reply"] != reply or item["action"] != calls[0]["action"]):
            failures.append("redelivery receipt mismatched")
    cost = before["state"]["spent"]
    previous_labels = before["labels"].copy()
    for event in events:
        kind, detail, state = event["event"], event["detail"], event["state"]
        if kind == "annotation_created":
            cost += 1
            if detail["sample"] in previous_labels:
                failures.append("duplicate sample charged")
            previous_labels[detail["sample"]] = detail["label"]
        elif kind == "annotation_revised":
            if previous_labels.get(detail["sample"]) != detail["old_label"]:
                failures.append("label revision lost history")
            previous_labels[detail["sample"]] = detail["label"]
        elif kind == "annotation_skipped" and detail["sample"] != before["state"].get("shown"):
            failures.append("skip was not linked to the presented sample")
        elif kind == "pause_requested" and state["phase"] not in {"training", "evaluating"}:
            failures.append("pause happened before the round boundary")
        elif kind == "task_paused" and state.get("last_evaluated_round") != state["current_round"]:
            failures.append("paused before evaluation")
        elif kind == "model_saved" and detail["checkpoint_sha256"] != after["checkpoints"].get(state["model"]):
            failures.append("checkpoint digest mismatch")
        if state["spent"] != cost or not 0 <= cost <= state["budget"]:
            failures.append("intermediate cost or budget invariant")
    if previous_labels != labels or actual["spent"] != len(labels) or not 0 <= actual["spent"] <= actual["budget"]:
        failures.append("persisted label/budget invariant")
    if actual["outcome"] == "applied" and kinds == ["job_enqueued"] and after["state"].get("failed_job") != before["state"].get("failed_job"):
        failures.append("retry lost original failed job")
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=ROOT / "research/experiments/任务样例.json")
    parser.add_argument("--out", type=Path, default=ROOT / "results/workflow")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cases = json.loads(args.cases.read_text("utf-8-sig"))["cases"]
    rows, traces = [], []
    for case in cases:
        with TemporaryDirectory() as temp:
            trace = run_actions(case["initial"], case["form_input"], case["advance"], Path(temp))
        failures = grade(trace, case["expected"], case["required_evidence"])
        if trace["live"] == trace["after"]:
            del trace["live"]
        traces.append(dict(case_id=case["id"], expected=case["expected"],
                           required_events=case["required_evidence"], failures=failures, **trace))
        before, after = trace["before"]["state"], trace["after"]["state"]
        rows.append(dict(case_id=case["id"], family=case["family"], split=case["split"],
                         scope=case["scope"], entry="structured", passed=int(not failures),
                         outcome=trace["calls"][0]["reply"]["outcome"], elapsed_ms=trace["elapsed_ms"],
                         event_count=len(trace["events"]), budget_before=before["budget"],
                         budget_after=after["budget"], spent_before=before["spent"],
                         spent_after=after["spent"], failures=" | ".join(failures)))
    with (args.out / "case_results.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.out / "traces.jsonl").write_text("".join(canonical(t) + "\n" for t in traces), "utf-8")

    def groups(field):
        buckets = sorted({r[field] for r in rows})
        return {k: {"passed": sum(r["passed"] for r in rows if r[field] == k),
                    "total": sum(r[field] == k for r in rows)} for k in buckets}

    summary = {
        "entry": "structured deterministic actions on isolated SQLite databases",
        "observation_unit": "one prewritten form-input case",
        "total": len(rows), "passed": sum(r["passed"] for r in rows),
        "split": groups("split"), "scope": groups("scope"), "family": groups("family"),
        "outcomes": dict(Counter(r["outcome"] for r in rows)),
        "source_sha256": {p.name: digest(p.read_bytes()) for p in (
            args.cases, Path(__file__).resolve(), Path(__file__).with_name("ledger.py"))},
        "limitations": ["Form inputs are pre-parsed; no language model or human subject in this run.",
                        "All initial labels, supplied checkpoints and worker callbacks are test fixtures.",
                        "Local elapsed time includes SQLite writes and scheduled callbacks; it is not user interaction time.",
                        "The case bank was visible during implementation; eval denotes a file partition, not a blind test."],
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", "utf-8")
    print(f'{summary["passed"]}/{summary["total"]} structured workflow cases passed')
    return int(summary["passed"] != len(rows))


if __name__ == "__main__":
    raise SystemExit(main())
