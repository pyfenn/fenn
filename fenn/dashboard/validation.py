"""Reusable validation helpers for dashboard API requests."""

from __future__ import annotations

from typing import Any

from flask import Response

from fenn.dashboard.responses import (
    invalid_param,
    invalid_request,
)

ValidationError = tuple[Response, int]


def check_body(
    payload: Any,
    expected_type: type[Any],
) -> ValidationError | None:
    """Validate the decoded request body type."""
    if not isinstance(payload, expected_type):
        return invalid_request(), 400

    return None


def check_non_empty_string(
    value: Any,
    param: str,
) -> ValidationError | None:
    """Validate that a parameter is a non-empty string."""
    if not isinstance(value, str) or not value.strip():
        return (
            invalid_param(
                message=f"{param} must be a non-empty string",
                param=param,
            ),
            400,
        )

    return None


def check_type(
    value: Any,
    expected_type: type[Any],
    param: str,
    expected_name: str | None = None,
) -> ValidationError | None:
    """Validate a parameter against an expected Python type."""
    if isinstance(value, expected_type):
        return None

    type_name = expected_name or expected_type.__name__

    return (
        invalid_param(
            message=f"{param} must be a {type_name}",
            param=param,
        ),
        400,
    )
