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
		action_columns = {row[1] for row in connection.execute("PRAGMA table_info(agent_actions)")}
		if "activity_type" in action_columns:
			connection.execute("ALTER TABLE agent_actions DROP COLUMN activity_type")
		connection.execute(
			"""
			CREATE TABLE IF NOT EXISTS internal_app_logs (
				id INTEGER PRIMARY KEY AUTOINCREMENT,
				order_reference TEXT NOT NULL,
				agent_action_id INTEGER NOT NULL UNIQUE,
				created_at TEXT NOT NULL,
				FOREIGN KEY(agent_action_id) REFERENCES agent_actions(id)
			)
			"""
		)
		connection.execute(
			"""
			CREATE TABLE IF NOT EXISTS manual_resolutions (
				id INTEGER PRIMARY KEY AUTOINCREMENT,
				order_reference TEXT NOT NULL,
				escalation_action_id INTEGER NOT NULL,
				outcome TEXT NOT NULL,
				selected_vehicle TEXT,
				courier_count INTEGER,
				explanation TEXT NOT NULL,
				verification_source TEXT,
				created_at TEXT NOT NULL
			)
			"""
		)
		connection.execute(
			"""
			CREATE TABLE IF NOT EXISTS internal_resolution_logs (
				id INTEGER PRIMARY KEY AUTOINCREMENT,
				order_reference TEXT NOT NULL,
				resolution_id INTEGER NOT NULL UNIQUE,
				created_at TEXT NOT NULL,
				FOREIGN KEY(resolution_id) REFERENCES manual_resolutions(id)
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
	"""Persist an agent action."""
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
			"SELECT id, order_reference, action, selected_vehicle, courier_count, reason, created_at "
			"FROM agent_actions ORDER BY id DESC",
			connection,
		)


def latest_agent_actions(database_path: str | Path) -> pd.DataFrame:
	"""Load only the latest saved action per order without modifying history."""
	actions = load_agent_actions(database_path)
	return actions.drop_duplicates("order_reference", keep="first").reset_index(drop=True)


def record_internal_app_log(database_path: str | Path, order_reference: str, agent_action_id: int) -> None:
	"""Append the agent's report that the latest confirmed/overridden decision was entered internally."""
	initialize_database(database_path)
	created_at = datetime.now(timezone.utc).isoformat()
	with sqlite3.connect(database_path) as connection:
		decision = connection.execute(
			"SELECT order_reference, action, selected_vehicle, courier_count FROM agent_actions WHERE id = ?",
			(agent_action_id,),
		).fetchone()
		if decision is None or str(decision[0]) != str(order_reference):
			raise ValueError("agent decision does not match this order")
		if decision[1] not in {"Confirm recommendation", "Override recommendation"} or not decision[2] or not decision[3]:
			raise ValueError("only a valid confirmed or overridden decision can be logged")
		latest = connection.execute(
			"SELECT id FROM agent_actions WHERE order_reference = ? ORDER BY id DESC LIMIT 1",
			(order_reference,),
		).fetchone()
		if latest is None or latest[0] != agent_action_id:
			raise ValueError("only the latest decision can be logged")
		try:
			connection.execute(
				"INSERT INTO internal_app_logs (order_reference, agent_action_id, created_at) VALUES (?, ?, ?)",
				(order_reference, agent_action_id, created_at),
			)
		except sqlite3.IntegrityError as error:
			raise ValueError("this decision has already been logged") from error


def load_internal_app_logs(database_path: str | Path) -> pd.DataFrame:
	"""Load append-only reports of decisions logged in the internal app."""
	initialize_database(database_path)
	with sqlite3.connect(database_path) as connection:
		return pd.read_sql_query(
			"SELECT id, order_reference, agent_action_id, created_at FROM internal_app_logs ORDER BY id DESC",
			connection,
		)


def record_manual_resolution(
	database_path: str | Path,
	order_reference: str,
	escalation_action_id: int,
	outcome: str,
	explanation: str,
	verification_source: str | None = None,
	selected_vehicle: str | None = None,
	courier_count: int | None = None,
) -> int:
	"""Append a human-verified resolution tied to the latest escalation."""
	if not explanation.strip():
		raise ValueError("verification explanation is required")
	if outcome not in {"delivery", "customer_intervention"}:
		raise ValueError("unsupported resolution outcome")
	if outcome == "delivery" and (not selected_vehicle or courier_count is None or courier_count < 1):
		raise ValueError("delivery resolution requires a vehicle and positive courier count")
	if outcome == "customer_intervention":
		selected_vehicle = None
		courier_count = None
	initialize_database(database_path)
	created_at = datetime.now(timezone.utc).isoformat()
	with sqlite3.connect(database_path) as connection:
		latest = connection.execute(
			"SELECT id, action FROM agent_actions WHERE order_reference = ? ORDER BY id DESC LIMIT 1",
			(order_reference,),
		).fetchone()
		if latest is None or latest[0] != escalation_action_id or latest[1] != "Escalate for review":
			raise ValueError("resolution must reference the latest escalation for this order")
		cursor = connection.execute(
			"""INSERT INTO manual_resolutions
			(order_reference, escalation_action_id, outcome, selected_vehicle, courier_count, explanation, verification_source, created_at)
			VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
			(order_reference, escalation_action_id, outcome, selected_vehicle, courier_count, explanation.strip(), verification_source, created_at),
		)
		return int(cursor.lastrowid)


def load_manual_resolutions(database_path: str | Path) -> pd.DataFrame:
	"""Load all append-only human verification resolutions."""
	initialize_database(database_path)
	with sqlite3.connect(database_path) as connection:
		return pd.read_sql_query("SELECT * FROM manual_resolutions ORDER BY id DESC", connection)


def record_internal_resolution_log(database_path: str | Path, order_reference: str, resolution_id: int) -> None:
	"""Record the agent's report that a human resolution was entered internally."""
	initialize_database(database_path)
	created_at = datetime.now(timezone.utc).isoformat()
	with sqlite3.connect(database_path) as connection:
		resolution = connection.execute(
			"SELECT order_reference, id, escalation_action_id FROM manual_resolutions WHERE id = ?",
			(resolution_id,),
		).fetchone()
		if resolution is None or str(resolution[0]) != str(order_reference):
			raise ValueError("resolution does not match this order")
		latest_action = connection.execute(
			"SELECT id, action FROM agent_actions WHERE order_reference = ? ORDER BY id DESC LIMIT 1",
			(order_reference,),
		).fetchone()
		if latest_action is None or latest_action[0] != resolution[2] or latest_action[1] != "Escalate for review":
			raise ValueError("only the current escalation resolution can be logged")
		latest_resolution = connection.execute(
			"SELECT id FROM manual_resolutions WHERE order_reference = ? ORDER BY id DESC LIMIT 1",
			(order_reference,),
		).fetchone()
		if latest_resolution is None or latest_resolution[0] != resolution_id:
			raise ValueError("only the latest resolution can be logged")
		try:
			connection.execute(
				"INSERT INTO internal_resolution_logs (order_reference, resolution_id, created_at) VALUES (?, ?, ?)",
				(order_reference, resolution_id, created_at),
			)
		except sqlite3.IntegrityError as error:
			raise ValueError("this resolution has already been logged") from error


def load_internal_resolution_logs(database_path: str | Path) -> pd.DataFrame:
	"""Load log acknowledgements for human resolutions."""
	initialize_database(database_path)
	with sqlite3.connect(database_path) as connection:
		return pd.read_sql_query("SELECT id, order_reference, resolution_id, created_at FROM internal_resolution_logs ORDER BY id DESC", connection)


def record_internal_app_log(database_path: str | Path, order_reference: str, agent_action_id: int) -> None:
	"""Record the agent's report that the latest valid decision was applied internally."""
	initialize_database(database_path)
	created_at = datetime.now(timezone.utc).isoformat()
	with sqlite3.connect(database_path) as connection:
		decision = connection.execute(
			"SELECT order_reference, action, selected_vehicle, courier_count "
			"FROM agent_actions WHERE id = ?",
			(agent_action_id,),
		).fetchone()
		if decision is None or str(decision[0]) != str(order_reference):
			raise ValueError("decision action does not match this order")
		if decision[1] not in {"Confirm recommendation", "Override recommendation"} or not decision[2] or not decision[3] or decision[3] < 1:
			raise ValueError("only a valid confirmed or overridden delivery decision can be logged")
		latest = connection.execute(
			"SELECT id FROM agent_actions WHERE order_reference = ? ORDER BY id DESC LIMIT 1",
			(order_reference,),
		).fetchone()
		if latest is None or latest[0] != agent_action_id:
			raise ValueError("only the latest order decision can be logged")
		try:
			connection.execute(
				"INSERT INTO internal_app_logs (order_reference, agent_action_id, created_at) VALUES (?, ?, ?)",
				(order_reference, agent_action_id, created_at),
			)
		except sqlite3.IntegrityError as error:
			raise ValueError("this decision has already been logged") from error