import re
import sys

COMMIT_MSG_FILE = sys.argv[1]

PATTERN = re.compile(
    r"^(feat|fix|docs|style|refactor|perf|test|build|ci|chore|revert)"
    r"(\(.+\))?!?: \[\#[0-9]+\] .+"
)

TYPES = "feat, fix, docs, style, refactor, perf, test, build, ci, chore, revert"

USAGE = f"""\
Commit message must follow the Conventional Commits format:

  <type>: [#<issue-id>] <description>

Supported types: {TYPES}
With optional scope and/or breaking change indicator:

  <type>(<scope>):   [#<issue-id>] <description>
  <type>!:           [#<issue-id>] <description>
  <type>(<scope>)!:  [#<issue-id>] <description>

Examples:
  fix: [#178] Add commit message check in pre-commit
  feat(dashboard): [#180] Add new widget
  docs!: [#181] Update installation guide
"""

with open(COMMIT_MSG_FILE) as f:
    lines = f.readlines()

commit_msg = "".join(lines).strip()

if not commit_msg:
    print("Commit message is empty.", file=sys.stderr)
    sys.exit(1)

first_line = commit_msg.split("\n")[0]

if first_line.startswith("Merge"):
    sys.exit(0)

if PATTERN.match(first_line):
    sys.exit(0)
else:
    print("ERROR: Invalid commit message format.\n", file=sys.stderr)
    print(USAGE, file=sys.stderr)
    print(f"Got: {first_line}", file=sys.stderr)
    sys.exit(1)
