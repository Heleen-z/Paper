"""One-task SQLite store for a budgeted adaptation session (standard library only)."""
import hashlib
import json
import sqlite3
from pathlib import Path


def canonical(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def digest(data):
    return hashlib.sha256(data).hexdigest()


class Refused(ValueError):
    def __init__(self, reason, *, missing=None):
        self.reply = ({"outcome": "needs_input", "missing": missing} if missing
                      else {"outcome": "rejected", "reason": reason})
        super().__init__(reason)


class Ledger:
    """Use one database per task. All state mutations share a single immediate transaction."""
    def __init__(self, path):
        self.db = sqlite3.connect(path, timeout=15, isolation_level=None)
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS task (id INTEGER PRIMARY KEY CHECK(id=1), value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS labels (sample TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS snapshots (name TEXT PRIMARY KEY, content TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS checkpoints (seq INTEGER PRIMARY KEY, name TEXT UNIQUE, payload BLOB NOT NULL, sha256 TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS events (seq INTEGER PRIMARY KEY, kind TEXT NOT NULL, detail TEXT NOT NULL, state TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS requests (key TEXT PRIMARY KEY, action TEXT NOT NULL, reply TEXT NOT NULL);
        """)
        self.db.execute("INSERT OR IGNORE INTO task VALUES(1,?)", (canonical({
            "phase": "absent", "budget": 0, "current_round": 0, "task_count": 0,
            "source": None, "target": None, "model": None, "snapshot": None,
            "active_job": None, "pause_requested": False}),))

    def state(self):
        state = json.loads(self.db.execute("SELECT value FROM task").fetchone()[0])
        state["spent"] = self.db.execute("SELECT count(*) FROM labels").fetchone()[0]
        return state

    def labels(self):
        return dict(self.db.execute("SELECT sample,value FROM labels ORDER BY sample"))

    def snapshot_labels(self, name):
        row = self.db.execute("SELECT content FROM snapshots WHERE name=?", (name,)).fetchone()
        if not row:
            raise ValueError("missing_label_snapshot")
        return json.loads(row[0])

    def checkpoint_bytes(self, name):
        row = self.db.execute("SELECT payload,sha256 FROM checkpoints WHERE name=?", (name,)).fetchone()
        if not row or not row[0] or digest(row[0]) != row[1]:
            raise ValueError("missing_or_invalid_checkpoint")
        return row[0]

    def events(self):
        return [{"seq": seq, "event": kind, "detail": json.loads(detail), "state": json.loads(state)}
                for seq, kind, detail, state in self.db.execute("SELECT * FROM events ORDER BY seq")]

    def _save(self, state):
        self.db.execute("UPDATE task SET value=? WHERE id=1", (canonical(
            {k: v for k, v in state.items() if k != "spent"}),))

    def _event(self, kind, detail, state):
        self._save(state)
        actual = self.state()
        if not 0 <= actual["spent"] <= actual["budget"]:
            raise RuntimeError("budget invariant violated")
        self.db.execute("INSERT INTO events(kind,detail,state) VALUES(?,?,?)",
                        (kind, canonical(detail), canonical(actual)))

    def _snapshot(self, name=None):
        content = canonical(self.labels())
        name = name or digest(content.encode())
        self.db.execute("INSERT OR IGNORE INTO snapshots VALUES(?,?)", (name, content))
        return name

    def _checkpoint(self, name, payload):
        self.db.execute("INSERT INTO checkpoints(name,payload,sha256) VALUES(?,?,?)",
                        (name, payload, digest(payload)))

    def _valid_model(self, name):
        row = self.db.execute("SELECT payload,sha256 FROM checkpoints WHERE name=?", (name,)).fetchone()
        return bool(row and row[0] and digest(row[0]) == row[1])

    def execute(self, action):
        if not isinstance(action, dict):
            return {"outcome": "rejected", "reason": "invalid_action"}
        request = action.get("request_id")
        if request is not None and (not isinstance(request, str) or not request.strip()):
            return {"outcome": "rejected", "reason": "invalid_request_id"}
        steps = action.get("ops", [action])
        if not isinstance(steps, list) or not steps or not all(isinstance(step, dict) for step in steps):
            return {"outcome": "rejected", "reason": "invalid_steps"}
        self.db.execute("BEGIN IMMEDIATE")
        try:
            if request:
                old = self.db.execute("SELECT action,reply FROM requests WHERE key=?", (request,)).fetchone()
                if old:
                    if canonical(action) != old[0]:
                        raise Refused("request_id_conflict")
                    self.db.commit()
                    return json.loads(old[1])
            for step in steps:
                reply = self._step(step)
            if request:
                self.db.execute("INSERT INTO requests VALUES(?,?,?)", (request, canonical(action), canonical(reply)))
            self.db.commit()
            return reply
        except Refused as error:
            self.db.rollback()
            return error.reply
        except Exception:
            self.db.rollback()
            raise

    def _step(self, action):
        state = self.state()
        op = action.get("op")
        if not isinstance(op, str):
            raise Refused("invalid_operation")
        required_ids = {
            "show": ("sample",), "skip": ("sample",), "submit": ("sample",), "correct": ("sample",),
            "start_training": ("job",), "job_failed": ("job", "reason"),
            "model_saved": ("job", "model"), "rollback": ("to",),
            "retry_failed_job": ("job",), "ensure_retry": ("job",),
        }
        for key in required_ids.get(op, ()):
            if not isinstance(action.get(key), str) or not action[key].strip():
                raise Refused("identifier_required", missing=key)
        phase = state["phase"]
        detail = dict(action)
        if op == "create":
            if phase != "absent":
                raise Refused("task_exists")
            for domain in ["source", "target"]:
                if not action.get(domain):
                    raise Refused("domain_required", missing=domain)
                if action[domain] not in ("Art", "Clipart", "Product", "Real_World"):
                    raise Refused("unknown_domain")
            if action["source"] == action["target"]:
                raise Refused("same_domain")
            if type(action.get("budget")) is not int or action["budget"] < 0:
                raise Refused("invalid_budget")
            state.update(phase="query_ready", task_count=1, current_round=1,
                         source=action["source"], target=action["target"], budget=action["budget"])
            kind = "task_created"
        elif phase == "absent":
            raise Refused("missing_task")
        elif op in {"set_budget", "set_remaining", "add_budget"}:
            amount = action.get("amount")
            if type(amount) is not int or amount < 0:
                raise Refused("invalid_budget")
            new = amount + {"set_budget": 0, "set_remaining": state["spent"], "add_budget": state["budget"]}[op]
            if new < state["spent"]:
                raise Refused("below_spent")
            state["budget"] = new
            kind = "budget_changed"
        elif op == "show":
            if phase not in {"query_ready", "annotating"}:
                raise Refused("invalid_phase")
            if state.get("shown") and state["shown"] != action["sample"]:
                raise Refused("unresolved_feedback")
            if self.db.execute("SELECT 1 FROM labels WHERE sample=?", (action["sample"],)).fetchone():
                raise Refused("sample_already_labeled")
            state.update(shown=action["sample"], phase="annotating")
            kind = "annotation_shown"
        elif op in {"skip", "submit", "correct"}:
            if phase != "annotating":
                raise Refused("invalid_phase")
            sample = action["sample"]
            old = self.db.execute("SELECT value FROM labels WHERE sample=?", (sample,)).fetchone()
            if op != "correct" and state.get("shown") != sample:
                raise Refused("not_shown")
            if op == "skip":
                state.update(skipped_sample=sample, shown=None)
                kind = "annotation_skipped"
            else:
                if not isinstance(action.get("label"), str) or not action["label"].strip():
                    raise Refused("empty_label")
                if (op == "correct") != bool(old):
                    raise Refused("unknown_or_duplicate_label")
                if not old and state["spent"] == state["budget"]:
                    raise Refused("budget_exhausted")
                detail.update(old_label=old[0] if old else None)
                self.db.execute("INSERT OR REPLACE INTO labels VALUES(?,?)", (sample, action["label"]))
                state["snapshot"] = self._snapshot()
                if state.get("shown") == sample:
                    state["shown"] = None
                kind = "annotation_revised" if old else "annotation_created"
        elif op == "start_training":
            if phase != "annotating" or state.get("shown") or not state["spent"] or state["active_job"]:
                raise Refused("not_ready_for_training")
            state.update(active_job=action["job"], parent_job=None, phase="training")
            kind = "job_enqueued"
        elif op == "job_failed":
            if phase != "training" or action["job"] != state["active_job"]:
                raise Refused("job_mismatch")
            state.update(active_job=None, failed_job=action["job"], fault=action["reason"], phase="failed")
            kind = "job_failed"
        elif op == "pause_at_round_end":
            if phase not in {"training", "evaluating"}:
                raise Refused("invalid_phase")
            state["pause_requested"] = True
            kind = "pause_requested"
        elif op == "model_saved":
            if phase != "training" or action["job"] != state["active_job"]:
                raise Refused("job_mismatch")
            if not isinstance(action.get("checkpoint_path"), str) or not action["checkpoint_path"]:
                raise Refused("path_required", missing="checkpoint_path")
            if self.db.execute("SELECT 1 FROM checkpoints WHERE name=?", (action["model"],)).fetchone():
                raise Refused("checkpoint_name_exists")
            try:
                payload = Path(action["checkpoint_path"]).read_bytes()
            except (OSError, ValueError):
                raise Refused("checkpoint_unreadable") from None
            if not payload:
                raise Refused("empty_checkpoint")
            self._checkpoint(action["model"], payload)
            detail.pop("checkpoint_path")
            detail["checkpoint_sha256"] = digest(payload)
            state.update(model=action["model"], phase="evaluating", active_job=None)
            kind = "model_saved"
        elif op == "evaluation_completed":
            if phase != "evaluating" or not self._valid_model(state["model"]):
                raise Refused("invalid_phase_or_checkpoint")
            state["last_evaluated_round"] = state["current_round"]
            if state["pause_requested"]:
                self._event("evaluation_completed", detail, state)
                state.update(phase="paused", pause_requested=False)
                kind = "task_paused"
            else:
                state.update(phase="query_ready", current_round=state["current_round"] + 1)
                kind = "evaluation_completed"
        elif op == "resume":
            if phase != "paused" or state.get("last_evaluated_round") != state["current_round"] or not self._valid_model(state["model"]):
                raise Refused("no_complete_round")
            state.update(phase="query_ready", current_round=state["current_round"] + 1)
            kind = "task_resumed"
        elif op in {"retry_failed_job", "ensure_retry"}:
            job = action["job"]
            if phase == "training" and job == state.get("parent_job"):
                return {"outcome": "already_running"}
            if phase != "failed" or job != state.get("failed_job"):
                raise Refused("no_failed_job")
            if not self.db.execute("SELECT 1 FROM snapshots WHERE name=?", (state["snapshot"],)).fetchone():
                raise Refused("missing_label_snapshot")
            state.update(phase="training", active_job=job + "_retry", parent_job=job)
            kind = "job_enqueued"
        elif op == "rollback":
            if phase != "paused":
                raise Refused("invalid_phase")
            target = action["to"]
            if target == "previous":
                row = self.db.execute("SELECT name FROM checkpoints WHERE seq < (SELECT seq FROM checkpoints WHERE name=?) ORDER BY seq DESC LIMIT 1",(state["model"],)).fetchone()
                target = row[0] if row else None
            if not self._valid_model(target):
                raise Refused("missing_checkpoint")
            state["model"] = target
            kind = "active_model_changed"
        else:
            raise Refused("unknown_operation")
        self._event(kind, detail, state)
        return {"outcome": "applied"}
