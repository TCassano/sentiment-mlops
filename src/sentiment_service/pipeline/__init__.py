"""Offline pipeline steps (data collection, evaluation, drift, retraining).

Plain Python functions: Airflow only orchestrates them, so every step can be run
and unit-tested without an Airflow installation.
"""
