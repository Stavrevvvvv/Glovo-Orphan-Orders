"""SQLite persistence for dashboard agent actions."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def initialize_database(database_path: str | Path) -> None:
	"""Create the action table if it does not already exist."""
	Path(database_path).parent.mkdir(parents=True, exist_ok=True)
	with sqlite3.connect(database_path) as connection:
		connection.execute(
			"""
			CREATE TABLE IF NOT EXISTS agent_actions (
				id INTEGER PRIMARY KEY AUTOINCREMENT,
				order_reference TEXT NOT NULL,
				action TEXT NOT NULL,
				selected_vehicle TEXT,
				courier_count INTEGER,
				reason TEXT,
				created_at TEXT NOT NULL
			)
			"""
		)


def record_agent_action(
	database_path: str | Path,
	order_reference: str,
	action: str,
	selected_vehicle: str | None,
	courier_count: int | None,
	reason: str,
) -> None:
	"""Persist one confirmation, override, or escalation event."""
	initialize_database(database_path)
	created_at = datetime.now(timezone.utc).isoformat()
	with sqlite3.connect(database_path) as connection:
		connection.execute(
			"""
			INSERT INTO agent_actions
			(order_reference, action, selected_vehicle, courier_count, reason, created_at)
			VALUES (?, ?, ?, ?, ?, ?)
			""",
			(order_reference, action, selected_vehicle, courier_count, reason, created_at),
		)


def load_agent_actions(database_path: str | Path) -> pd.DataFrame:
	"""Load persisted agent actions, returning an empty typed frame when absent."""
	initialize_database(database_path)
	with sqlite3.connect(database_path) as connection:
		return pd.read_sql_query(
			"SELECT order_reference, action, selected_vehicle, courier_count, reason, created_at "
			"FROM agent_actions ORDER BY created_at DESC",
			connection,
		)