"""Reusable JSON responses for dashboard API endpoints."""

from __future__ import annotations

from flask import Response, jsonify


def error_response(
    code: str,
    message: str,
    param: str | None = None,
) -> Response:
    """Create the standard dashboard API error envelope."""
    error = {
        "code": code,
        "message": message,
    }

    if param is not None:
        error["param"] = param

    return jsonify({"error": error})


def invalid_request(
    message: str = "A JSON request body is required",
) -> Response:
    return error_response(
        code="invalid_request",
        message=message,
    )


def invalid_param(
    message: str,
    param: str,
) -> Response:
    return error_response(
        code="invalid_param",
        message=message,
        param=param,
    )


def template_list_unavailable(exc: Exception) -> Response:
    return error_response(
        code="template_list_unavailable",
        message=str(exc),
    )


def target_not_empty(exc: Exception) -> Response:
    return error_response(
        code="target_not_empty",
        message=str(exc),
        param="path",
    )


def template_not_found(exc: Exception) -> Response:
    return error_response(
        code="template_not_found",
        message=str(exc),
        param="template",
    )


def template_download_unavailable(exc: Exception) -> Response:
    return error_response(
        code="template_download_unavailable",
        message=str(exc),
    )


def invalid_template(exc: Exception) -> Response:
    return error_response(
        code="invalid_template",
        message=str(exc),
    )


def filesystem_error(exc: Exception) -> Response:
    return error_response(
        code="filesystem_error",
        message=str(exc),
        param="path",
    )
