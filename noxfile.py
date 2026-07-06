import nox

nox.options.default_venv_backend = "uv"
nox.options.reuse_existing_virtualenvs = True

EDITABLE_INSTALL = ("uv", "sync", "--group", "test")


@nox.session(tags=["fenn"])
def unit(session: nox.Session) -> None:
    session.run_install(*EDITABLE_INSTALL)
    session.run("pytest", "tests/unit", "--cov", "--cov-branch", "--cov-report=xml")
