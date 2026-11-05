"""Weekly loop: collect labelled posts, evaluate the production model, check drift and,
if needed, fine-tune a candidate and promote it only if it beats the current model.

    collect_data -> validate_data -> evaluate_current_model -> check_drift -> decide
        decide --(retrain)--> retrain_model -> evaluate_and_promote --+
        decide --(ok)-------> skip_retraining ------------------------+--> publish_report

All the logic lives in ``sentiment_service.pipeline.steps`` (unit-tested without
Airflow); this file only wires the steps together. Requires the package to be installed
in the Airflow workers (``pip install .`` from the repo root).
"""

from __future__ import annotations

from datetime import datetime, timedelta

try:  # Airflow 3
    from airflow.sdk import dag, task
except ImportError:  # Airflow 2.x
    from airflow.decorators import dag, task

DEFAULT_ARGS = {
    "owner": "applied-ml",
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
}


@dag(
    dag_id="sentiment_monitoring_retraining",
    schedule="@weekly",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    default_args=DEFAULT_ARGS,
    tags=["sentiment", "mlops"],
    doc_md=__doc__,
)
def sentiment_monitoring_retraining():
    @task
    def collect_data(run_id: str) -> str:
        from sentiment_service.pipeline import steps

        return steps.collect(run_id)

    @task
    def validate_data(batch_path: str) -> str:
        from sentiment_service.pipeline import steps

        return steps.validate(batch_path)

    @task
    def evaluate_current_model(batch_path: str, run_id: str) -> dict:
        from sentiment_service.pipeline import steps

        return steps.evaluate_current(batch_path, run_id)

    @task
    def check_drift(eval_result: dict) -> dict:
        from sentiment_service.pipeline import steps

        return steps.check_drift(eval_result["predictions_path"])

    @task.branch
    def decide(eval_result: dict, drift_result: dict) -> str:
        from sentiment_service.pipeline import steps

        decision = steps.decide(eval_result, drift_result)
        print("Decision:", decision)
        return "retrain_model" if decision["retrain"] else "skip_retraining"

    @task(execution_timeout=timedelta(hours=3))
    def retrain_model(batch_path: str, run_id: str) -> str:
        from sentiment_service.pipeline import steps

        return steps.retrain(batch_path, run_id)

    @task
    def evaluate_and_promote(candidate: str, run_id: str) -> dict:
        from sentiment_service.pipeline import steps

        return steps.promote_or_reject(candidate, run_id)

    @task
    def skip_retraining() -> None:
        print("Model healthy and no drift: nothing to retrain.")

    @task(trigger_rule="none_failed_min_one_success")
    def publish_report(eval_result: dict, drift_result: dict, run_id: str) -> str:
        from sentiment_service.pipeline import steps

        return steps.write_report(run_id, evaluation=eval_result, drift=drift_result)

    run_id = "{{ ds_nodash }}"
    batch = validate_data(collect_data(run_id))
    evaluated = evaluate_current_model(batch, run_id)
    drifted = check_drift(evaluated)
    branch = decide(evaluated, drifted)

    candidate = retrain_model(batch, run_id)
    promoted = evaluate_and_promote(candidate, run_id)
    skipped = skip_retraining()
    branch >> [candidate, skipped]

    report = publish_report(evaluated, drifted, run_id)
    [promoted, skipped] >> report


sentiment_monitoring_retraining()
