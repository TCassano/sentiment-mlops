"""DAG integrity test: runs only where Airflow is installed (e.g. the Airflow CI image)."""

from pathlib import Path

import pytest

DAG_FOLDER = Path(__file__).resolve().parents[1] / "airflow" / "dags"


def test_dag_loads_and_has_expected_structure():
    # "airflow.models", not "airflow": the repo's airflow/ folder is importable as a namespace.
    DagBag = pytest.importorskip("airflow.models").DagBag  # noqa: N806

    bag = DagBag(dag_folder=str(DAG_FOLDER), include_examples=False)
    assert bag.import_errors == {}
    dag = bag.get_dag("sentiment_monitoring_retraining")
    assert dag is not None
    assert set(dag.task_dict) == {
        "collect_data",
        "validate_data",
        "evaluate_current_model",
        "check_drift",
        "decide",
        "retrain_model",
        "evaluate_and_promote",
        "skip_retraining",
        "publish_report",
    }
    assert dag.get_task("decide").downstream_task_ids == {"retrain_model", "skip_retraining"}


def test_dag_file_is_valid_python():
    source = (DAG_FOLDER / "sentiment_retraining_dag.py").read_text(encoding="utf-8")
    compile(source, "sentiment_retraining_dag.py", "exec")
