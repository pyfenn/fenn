"""fenn dashboard — launch the Fenn log-browser web UI."""

from fenn.logging import logger

# Must remain 127.0.0.1: the dashboard serves the user's own logs without any
# network-layer auth, and the in-process auth gate assumes a local-only socket.
# Do NOT bind to 0.0.0.0 or a non-loopback address.
DASHBOARD_HOST = "127.0.0.1"


def execute(args) -> None:
    """Import and start the dashboard server directly from the installed package."""
    try:
        from fenn.dashboard.app import run
    except ImportError as exc:
        raise SystemExit(
            "ERROR: Could not import the Fenn dashboard.\n"
            "Make sure Flask is installed:  pip install flask\n"
            f"Details: {exc}"
        )

    try:
        run(
            host=DASHBOARD_HOST,
            port=args.port,
            debug=args.debug,
            log_dirs=args.log_dir,
        )
    except KeyboardInterrupt:
        logger.info("\nDashboard stopped.")
