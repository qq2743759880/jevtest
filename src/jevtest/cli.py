"""jevtest 命令行入口。子命令：run（T12 单条分诊）/ benchmark（T09）/ synthgen（T04）。

run 子命令的退出码约定：
- 0  成功（含 E_PARSE_FAILURE 兜底路径——兜底结果如实标注，不视为进程失败）；
- 2  E_SCHEMA_INVALID_INPUT：输入文件缺失 / 非 JSON / 违反 FailureState 契约；
- 4  E_PROVIDER_AUTH：arm=jev/llm 需要 API key，key 未配置（.env / 环境变量）；
- 5  其他运行期错误（一行报错，不输出崩溃栈）。
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

_ARM_CHOICES = ("rule", "jev", "llm", "jev-mock", "cascade")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jevtest",
        description="JevTest Triage: triage network-device automation test failures",
    )
    sub = parser.add_subparsers(dest="command")
    run = sub.add_parser(
        "run", help="triage a single failure record (FailureState JSON) and print the three-question verdict"
    )
    run.add_argument("failure", help="path to a single FailureState JSON file (contracts/failure-state.schema.json)")
    run.add_argument(
        "--arm",
        choices=_ARM_CHOICES,
        default="rule",
        help="triage arm: rule (local rules, no key), jev (TypeSafe API), "
        "llm (Volcano Ark API), jev-mock (deterministic mock, no network), "
        "cascade (explicit offline RoutingPolicyV1). Default: rule",
    )
    run.add_argument(
        "--out",
        default=None,
        metavar="PATH",
        help="write the result JSON to PATH (legacy TriageResult or Cascade envelope)",
    )
    run.add_argument(
        "--policy",
        default=None,
        metavar="PATH",
        help="explicit RoutingPolicyV1 YAML (required for --arm cascade)",
    )
    run.add_argument(
        "--report",
        default=None,
        metavar="PATH",
        help="write a RunReportV2 JSON (Cascade defaults to a sibling report path)",
    )
    run.add_argument(
        "--local-store",
        default=None,
        metavar="PATH",
        help="persist Cascade results and queue required human review in a local SQLite store",
    )
    run.add_argument(
        "--idempotency-key",
        default=None,
        help="required with --local-store; reuse the key to safely retry the same handoff",
    )
    run.add_argument(
        "--target-role",
        default="human-review",
        help="role label for a local human-review handoff (not a named recipient)",
    )
    benchmark = sub.add_parser(
        "benchmark", help="run the multi-arm benchmark on a dataset (T09)"
    )
    benchmark.add_argument("dataset", help="dataset JSONL path (rows = {expected, failure_state})")
    benchmark.add_argument(
        "--arms",
        default="rule,jev,llm",
        help="comma-separated arm subset of rule/jev/llm/cascade (default: rule,jev,llm)",
    )
    benchmark.add_argument(
        "--limit", type=int, default=None, help="sample size (see --sampling for row selection)"
    )
    benchmark.add_argument(
        "--sampling",
        choices=["first", "random", "stratified"],
        default="first",
        help="row selection: first=file order (smoke); random=seeded sample; "
        "stratified=per-category quota (required for calibration/official runs — "
        "this dataset is category-blocked, 'first' takes a single-category block)",
    )
    benchmark.add_argument("--seed", type=int, default=0, help="seed for random/stratified sampling")
    benchmark.add_argument("--out", default="runs/benchmark", help="output directory")
    review = sub.add_parser("review", help="record a local interactive review of a stored result")
    review.add_argument("request_id", help="request id returned by a Cascade run")
    review.add_argument("--store", required=True, metavar="PATH", help="local SQLite store path")
    review.add_argument("--result-version", default=None, help="review a specific current result version")
    review.add_argument("--decision", choices=["accept", "correct", "defer"], required=True)
    review.add_argument("--reason", default=None, help="reason for correction or deferral")
    review.add_argument("--after-result", default=None, metavar="PATH", help="corrected TriageResult JSON")
    synthgen = sub.add_parser("synthgen", help="generate the synthetic failure dataset (T04)")
    synthgen.add_argument("--seed", type=int, default=20260924, help="random seed (default 20260924)")
    synthgen.add_argument("--out", default="data/synthetic/v0.jsonl", help="output JSONL path")
    synthgen.add_argument("--per-category", type=int, default=30, help="records per category")
    synthgen.add_argument("--unknown-count", type=int, default=20, help="extra UNKNOWN boundary records")
    return parser


class _ProviderAuthError(Exception):
    """arm=jev/llm 的 API key 未配置（E_PROVIDER_AUTH，exit 4）。"""


def _load_env_file() -> None:
    """沿用 jev_live 的 .env 加载机制（经模块属性调用，便于测试替换）。"""
    from jevtest.providers import jev_live

    jev_live._load_env_file()


def _build_provider(arm: str) -> tuple[object, str]:
    """按臂名构造 provider，返回 (provider, cost_state)。key 缺失抛 _ProviderAuthError。

    cost_state 与 benchmark 臂级口径一致：known=单价已冻结；unknown=token 已记账但
    无公开单价；mock=恒为 0。
    """
    if arm == "rule":
        from jevtest.providers.rule import RuleProvider

        return RuleProvider(), "known"
    if arm == "jev-mock":
        from jevtest.providers.jev_mock import JevMockProvider

        return JevMockProvider(), "mock"
    # 以下两臂需要 API key：先加载 .env（不覆盖已有环境变量）
    _load_env_file()
    if arm == "jev":
        from jevtest.providers.jev_live import JevLiveProvider

        try:
            return JevLiveProvider(), "known"
        except RuntimeError as exc:
            if "is not set" in str(exc):
                raise _ProviderAuthError(str(exc)) from exc
            raise
    if arm == "llm":
        from jevtest.providers.llm_judge import LLMJudgeConfigError, LLMJudgeProvider

        try:
            return LLMJudgeProvider(), "unknown"
        except LLMJudgeConfigError as exc:
            if "is not set" in str(exc):
                raise _ProviderAuthError(str(exc)) from exc
            raise
    raise ValueError(f"unknown arm: {arm!r}")  # argparse choices 已拦截，防御性保留


def _cmd_run(args: argparse.Namespace) -> int:
    from pydantic import ValidationError

    from jevtest.metrics import MeasureRunner
    from jevtest.models import FailureState, TriageResult
    from jevtest.rule_engine import _load_rule_contract

    # ---- 1. 读入并校验 FailureState ----
    path = Path(args.failure)
    try:
        raw_text = path.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"E_SCHEMA_INVALID_INPUT: cannot read failure record {path}: {exc}", file=sys.stderr)
        return 2
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        print(f"E_SCHEMA_INVALID_INPUT: {path} is not valid JSON: line {exc.lineno} column {exc.colno} ({exc.msg})", file=sys.stderr)
        return 2
    try:
        state = FailureState.model_validate(payload)
    except ValidationError as exc:
        fields = sorted({str(err["loc"][0]) for err in exc.errors() if err.get("loc")})
        details = "; ".join(
            f"{'.'.join(str(part) for part in err['loc'])}: {err['msg']}" for err in exc.errors()[:5]
        )
        field_note = f"fields: {', '.join(fields)}" if fields else "fields: (top level)"
        print(
            f"E_SCHEMA_INVALID_INPUT: failure record violates the FailureState contract "
            f"({field_note}): {details}",
            file=sys.stderr,
        )
        return 2

    if args.arm == "cascade":
        return _cmd_cascade(args, state)
    if (
        args.policy is not None
        or args.report is not None
        or args.local_store is not None
        or args.idempotency_key is not None
        or args.target_role != "human-review"
    ):
        print(
            "error: --policy, --report, --local-store, and --idempotency-key are available only with --arm cascade",
            file=sys.stderr,
        )
        return 2

    # ---- 2. 构造对应臂的 provider ----
    try:
        provider, cost_state = _build_provider(args.arm)
    except _ProviderAuthError as exc:
        print(f"E_PROVIDER_AUTH: {exc}", file=sys.stderr)
        print(
            "hint: put the API key in the project .env (TYPESAFE_API_KEY for --arm jev, "
            "ARK_API_KEY for --arm llm); --arm rule and --arm jev-mock need no key.",
            file=sys.stderr,
        )
        return 4
    except Exception as exc:  # 配置缺失等构造期错误：一行报错，不出崩溃栈
        print(f"error: failed to initialize arm {args.arm!r}: {exc}", file=sys.stderr)
        return 5

    # ---- 3. 计量执行（predict_validated 统一入口，契约兜底同 benchmark）----
    try:
        result, record = MeasureRunner(provider).run_one(state)
    except Exception as exc:
        print(f"error: arm {args.arm!r} failed on case {state.case_id!r}: {exc}", file=sys.stderr)
        return 5
    finally:
        close = getattr(provider, "close", None)
        if callable(close):
            close()

    # ---- 4. 人可读输出（三问 + reason + confidence）----
    rule_version = _load_rule_contract()["rule_version"]
    print(f"══ JevTest Triage ══  case: {state.case_id}")
    print(f"类别    : {result.category}")
    print(f"下一步  : {result.action}")
    print(f"优先级  : {result.priority}")
    print(f"置信度  : {result.confidence:.2f} (cost_state: {cost_state})")
    print(f"依据    : {result.reason}")
    if record.parse_failure:
        print(
            "警告    : E_PARSE_FAILURE — provider 输出违反 TriageResult 契约，已按 UNKNOWN+HUMAN_TRIAGE 兜底",
            file=sys.stderr,
        )
    print(f"── arm={args.arm}, rule_version={rule_version}, latency {record.latency_ms:.1f}ms")

    if args.out is not None:
        assert isinstance(result, TriageResult)
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(result.model_dump(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        print(f"triage result written to {out_path}")
    return 0


def _build_offline_cascade_providers() -> dict[str, object]:
    """Construct local Rule/Jev-mock/LLM-stub providers; this path has no live provider."""
    from jevtest.providers.jev_mock import JevMockProvider
    from jevtest.providers.llm_stub import OfflineLLMStubProvider
    from jevtest.providers.rule import RuleProvider

    return {
        "rule": RuleProvider(),
        "jev": JevMockProvider(),
        "llm": OfflineLLMStubProvider(),
    }


def _cmd_cascade(args: argparse.Namespace, state: object) -> int:
    from pydantic import ValidationError

    from jevtest.execution_contracts import RoutingPolicyV1
    from jevtest.models import FailureState
    from jevtest.run_reporting import build_run_report

    if not args.policy:
        print("E_POLICY_REQUIRED: --arm cascade requires --policy", file=sys.stderr)
        return 2
    if not args.out:
        print("E_OUTPUT_REQUIRED: --arm cascade requires --out for its envelope", file=sys.stderr)
        return 2
    if args.idempotency_key and not args.local_store:
        print("E_STORE_REQUIRED: --idempotency-key requires --local-store", file=sys.stderr)
        return 2
    if args.local_store and not args.idempotency_key:
        print("E_IDEMPOTENCY_REQUIRED: --local-store requires --idempotency-key", file=sys.stderr)
        return 2
    if not isinstance(state, FailureState):
        raise TypeError("_cmd_cascade requires a validated FailureState")

    policy_path = Path(args.policy)
    try:
        import yaml

        policy_payload = yaml.safe_load(policy_path.read_text(encoding="utf-8"))
        policy = RoutingPolicyV1.model_validate(policy_payload)
    except OSError as exc:
        print(f"E_POLICY_INVALID: cannot read policy {policy_path}: {exc}", file=sys.stderr)
        return 2
    except (ValidationError, ValueError, TypeError, yaml.YAMLError) as exc:
        print(f"E_POLICY_INVALID: {exc}", file=sys.stderr)
        return 2

    if policy.execution_mode != "offline":
        print("E_POLICY_UNAUTHORIZED: this Cascade CLI entry only supports offline policies", file=sys.stderr)
        return 2
    if policy.run_purpose in {"synthetic_eval", "formal_eval"}:
        print(
            "E_LABELS_REQUIRED: Cascade CLI inference does not accept evaluation claims without labels and evidence",
            file=sys.stderr,
        )
        return 2

    request_id = str(uuid4())
    result_version = str(uuid4())
    out_path = Path(args.out)
    report_path = Path(args.report) if args.report else out_path.with_name(
        f"{out_path.stem}.run-report-v2.json"
    )
    handoff_json_path = out_path.with_name(f"{out_path.stem}.handoff.json")
    handoff_markdown_path = out_path.with_name(f"{out_path.stem}.handoff.md")
    try:
        from jevtest.cascade import run_cascade

        envelope = run_cascade(
            state,
            policy,
            _build_offline_cascade_providers(),
            request_id=request_id,
            result_version=result_version,
        )
    except (ValidationError, ValueError, TypeError) as exc:
        print(f"E_CASCADE_INVALID: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"E_CASCADE_FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 5

    write_handoff_card = bool(args.local_store and envelope.requires_human_review)
    output_paths = [out_path, report_path]
    if write_handoff_card:
        output_paths.extend([handoff_json_path, handoff_markdown_path])
    resolved_outputs = [path.resolve() for path in output_paths]
    if len(set(resolved_outputs)) != len(resolved_outputs):
        print("E_OUTPUT_CONFLICT: result, report, and handoff card paths must be different", file=sys.stderr)
        return 2
    if args.local_store and Path(args.local_store).resolve() in resolved_outputs:
        print("E_OUTPUT_CONFLICT: --local-store must not overwrite an output file", file=sys.stderr)
        return 2

    if args.local_store:
        from jevtest.input_projection import project_failure_state
        from jevtest.local_store import LocalStore, LocalStoreConflictError

        try:
            with LocalStore(args.local_store) as store:
                facts = project_failure_state(state)
                if envelope.requires_human_review:
                    policy_scope = f"local-cli:{policy.policy_id}@{policy.version}"
                    envelope, handoff = store.queue_handoff(
                        envelope,
                        idempotency_scope=policy_scope,
                        idempotency_key=args.idempotency_key,
                        target_role=args.target_role,
                        common_facts=facts,
                    )
                    store.export_handoff_json(handoff.handoff_id, handoff_json_path)
                    store.export_handoff_markdown(handoff.handoff_id, handoff_markdown_path)
                else:
                    envelope = store.put_envelope(envelope)
                    store.put_common_facts(envelope.request_id, envelope.result_version, facts)
        except LocalStoreConflictError as exc:
            print(f"E_LOCAL_STORE_CONFLICT: {exc}", file=sys.stderr)
            return 2
        except (OSError, sqlite3.Error, ValidationError) as exc:
            print(
                f"E_LOCAL_STORE_FAILED: {type(exc).__name__}: {exc}; "
                "if the handoff was queued before a card export failure, retry with the same idempotency key",
                file=sys.stderr,
            )
            return 5

    try:
        report = build_run_report(
            [envelope],
            report_id=f"{envelope.request_id}-report",
            run_purpose=policy.run_purpose,
            dataset_kind="synthetic" if state.provenance.synthetic else "real_unlabeled",
            dataset_version=state.provenance.dataset_version,
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(envelope.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        report_path.write_text(
            json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    except (ValidationError, ValueError, TypeError) as exc:
        print(f"E_REPORT_INVALID: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"E_OUTPUT_WRITE_FAILED: {exc}", file=sys.stderr)
        return 5
    print(
        json.dumps(
            {
                "result": str(out_path),
                "report": str(report_path),
                "run_status": envelope.run_status,
                "category": envelope.result.category if envelope.result else None,
                "route": [step.stage for step in envelope.route_steps],
                "requires_human_review": envelope.requires_human_review,
                "delivery_status": envelope.delivery_status,
                "local_handoff_id": envelope.local_handoff_id,
                "handoff_json": str(handoff_json_path) if envelope.delivery_status == "local_queued" else None,
                "handoff_markdown": (
                    str(handoff_markdown_path) if envelope.delivery_status == "local_queued" else None
                ),
                "accuracy": report.accuracy,
            },
            ensure_ascii=False,
        )
    )
    return 0


def _cmd_review(args: argparse.Namespace) -> int:
    from pydantic import ValidationError

    if not sys.stdin.isatty():
        print("E_REVIEW_INTERACTIVE_REQUIRED: local review requires an interactive terminal", file=sys.stderr)
        return 2
    if args.store != ":memory:" and not Path(args.store).is_file():
        print("E_REVIEW_STORE_NOT_FOUND: local review store does not exist", file=sys.stderr)
        return 2

    from jevtest.execution_contracts import ReviewRecordV1
    from jevtest.local_store import LocalStore, LocalStoreConflictError
    from jevtest.models import TriageResult
    if args.decision == "correct" and not args.after_result:
        print("E_REVIEW_CORRECTION_REQUIRED: correct requires --after-result", file=sys.stderr)
        return 2
    if args.decision != "correct" and args.after_result:
        print("E_REVIEW_CORRECTION_UNEXPECTED: --after-result is valid only for correct", file=sys.stderr)
        return 2
    if args.decision in {"correct", "defer"} and not (args.reason and args.reason.strip()):
        print(f"E_REVIEW_REASON_REQUIRED: {args.decision} requires --reason", file=sys.stderr)
        return 2

    after_result = None
    if args.after_result:
        try:
            after_payload = json.loads(Path(args.after_result).read_text(encoding="utf-8"))
            after_result = TriageResult.model_validate(after_payload)
        except OSError as exc:
            print(f"E_REVIEW_CORRECTION_INVALID: cannot read {args.after_result}: {exc}", file=sys.stderr)
            return 2
        except (json.JSONDecodeError, ValidationError) as exc:
            print(f"E_REVIEW_CORRECTION_INVALID: {exc}", file=sys.stderr)
            return 2

    try:
        with LocalStore(args.store) as store:
            envelope = store.get_envelope(args.request_id, args.result_version)
            if envelope is None or envelope.result is None:
                print("E_REVIEW_RESULT_NOT_FOUND: request/result version has no stored result", file=sys.stderr)
                return 2
            current = store.get_envelope(args.request_id)
            if current is None or current.result_version != envelope.result_version:
                print("E_REVIEW_STALE_RESULT: review the current result version", file=sys.stderr)
                return 2
            if args.result_version is None and current.result_version != envelope.result_version:
                print("E_REVIEW_STALE_RESULT: review the current result version", file=sys.stderr)
                return 2

            print(
                f"Reviewing {envelope.request_id}@{envelope.result_version}: "
                f"{envelope.result.category} / {envelope.result.action}"
            )
            print("actor_origin is an asserted label; this local CLI does not authenticate identity.")
            try:
                actor_origin = input("Operator label: ").strip()
                confirmation = f"{args.decision} {args.request_id[:8]}"
                typed = input(f"Type '{confirmation}' to confirm: ").strip()
            except EOFError:
                print("E_REVIEW_CANCELLED: input ended; no review event was written", file=sys.stderr)
                return 2
            if not actor_origin:
                print("E_REVIEW_ACTOR_REQUIRED: operator label cannot be empty", file=sys.stderr)
                return 2
            if typed != confirmation:
                print("E_REVIEW_CONFIRMATION_MISMATCH: no review event was written", file=sys.stderr)
                return 2

            review = ReviewRecordV1(
                schema_version="review-record-v1",
                review_id=str(uuid4()),
                request_id=envelope.request_id,
                result_version=envelope.result_version,
                actor_type="local_user",
                actor_origin=actor_origin,
                assistance_mode="model_assisted",
                decision=args.decision,
                before_result=envelope.result,
                after_result=after_result,
                reason=args.reason,
                reviewed_at=datetime.now(timezone.utc).isoformat(),
            )
            stored_review = store.append_review(review)
    except LocalStoreConflictError as exc:
        print(f"E_REVIEW_CONFLICT: {exc}", file=sys.stderr)
        return 2
    except (ValidationError, ValueError, OSError, sqlite3.Error) as exc:
        print(f"E_REVIEW_INVALID: {exc}", file=sys.stderr)
        return 2

    print(
        json.dumps(
            {
                "review_id": stored_review.review_id,
                "request_id": stored_review.request_id,
                "result_version": stored_review.result_version,
                "decision": stored_review.decision,
            },
            ensure_ascii=False,
        )
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        return _cmd_run(args)
    if args.command == "review":
        return _cmd_review(args)
    if args.command == "synthgen":
        from jevtest.synthgen import generate_dataset, summarize

        path = generate_dataset(args.seed, args.out, args.per_category, args.unknown_count)
        print(json.dumps({"path": str(path), **summarize(path)}, ensure_ascii=False))
    elif args.command == "benchmark":
        from jevtest.benchmark import (
            EvaluationNotEligibleError,
            InputParityError,
            UnknownArmError,
            render_report,
            run_benchmark,
        )

        arms = [arm.strip() for arm in args.arms.split(",") if arm.strip()]
        try:
            report_path = run_benchmark(
                args.dataset, arms, limit=args.limit, sampling=args.sampling, seed=args.seed, out_dir=args.out
            )
        except EvaluationNotEligibleError as exc:
            print(f"error: evaluation not eligible: {exc}")
            return 2
        except UnknownArmError as exc:
            print(f"error: {exc}")
            return 2
        except InputParityError as exc:
            print(f"error: input parity violation: {exc}")
            return 3
        md_text = render_report(report_path)
        print(
            json.dumps(
                {
                    "report": str(report_path),
                    "markdown": str(report_path.parent / "experiment-report.md"),
                    "arms": json.loads(report_path.read_text(encoding="utf-8"))["arms"],
                    "markdown_chars": len(md_text),
                },
                ensure_ascii=False,
            )
        )
    elif args.command:
        print(f"jevtest {args.command}: unknown subcommand")
    return 0
