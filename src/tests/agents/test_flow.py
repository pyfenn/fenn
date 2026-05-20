import warnings
import pytest
from fenn.agents import BaseNode, Node, Flow


class _Counter(Node):
    """Node that increments shared['count'] and returns next_action."""
    def __init__(self, next_action="default"):
        super().__init__()
        self._next_action = next_action

    def exec(self, prep_res):
        return self._next_action

    def post(self, shared, prep_res, exec_res):
        shared["count"] = shared.get("count", 0) + 1
        return exec_res


# ── Linear flow ────────────────────────────────────────────────────────────────

def test_linear_flow_runs_all_nodes():
    a, b, c = _Counter(), _Counter(), _Counter()
    flow = Flow(start=a)
    flow.connect(a, b).connect(b, c).connect(c, None)

    shared = {}
    flow.run(shared)
    assert shared["count"] == 3


def test_linear_flow_preserves_order():
    order = []

    class _Ordered(Node):
        def __init__(self, name, nxt="default"):
            super().__init__()
            self.name = name
            self._nxt = nxt
        def post(self, shared, p, e):
            order.append(self.name)
            return self._nxt

    a, b = _Ordered("a"), _Ordered("b")
    flow = Flow(start=a)
    flow.connect(a, b).connect(b, None)
    flow.run({})
    assert order == ["a", "b"]


# ── Branching ──────────────────────────────────────────────────────────────────

def test_branching_by_action():
    class _Branch(Node):
        def __init__(self, action):
            super().__init__()
            self._action = action
        def post(self, shared, p, e):
            return self._action

    root    = _Branch("go_left")
    left    = _Counter()
    right   = _Counter()

    flow = Flow(start=root)
    flow.connect(root, left,  "go_left")
    flow.connect(root, right, "go_right")
    flow.connect(left,  None)
    flow.connect(right, None)

    shared = {}
    flow.run(shared)
    assert shared.get("count") == 1  # only left ran


# ── Explicit terminal transitions ──────────────────────────────────────────────

def test_explicit_terminal_no_warning():
    node = _Counter()
    flow = Flow(start=node)
    flow.connect(node, None)  # terminal

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        flow.run({})
    flow_warnings = [x for x in w if "Flow ends" in str(x.message)]
    assert not flow_warnings, "explicit terminal must not warn"


def test_missing_action_warns():
    class _WrongAction(Node):
        def post(self, shared, p, e): return "nonexistent"

    node = _WrongAction()
    flow = Flow(start=node)
    flow.connect(node, _Counter(), "default")  # only "default" is wired

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        flow.run({})
    assert any("Flow ends" in str(x.message) for x in w)


# ── Duplicate connection warning ───────────────────────────────────────────────

def test_duplicate_connect_warns():
    a, b, c = _Counter(), _Counter(), _Counter()
    flow = Flow(start=a)
    flow.connect(a, b)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        flow.connect(a, c)  # overwrite
    assert any("Overwriting successor" in str(x.message) for x in w)


# ── Old operator wiring is gone ────────────────────────────────────────────────

def test_no_rshift_operator():
    assert not hasattr(BaseNode, "__rshift__"), "BaseNode.__rshift__ must be removed"


def test_no_sub_operator():
    assert not hasattr(BaseNode, "__sub__"), "BaseNode.__sub__ must be removed"


def test_no_next_method():
    assert not hasattr(BaseNode, "next"), "BaseNode.next must be removed"


# ── Flow.connect returns self ──────────────────────────────────────────────────

def test_connect_returns_flow():
    a, b = _Counter(), _Counter()
    flow = Flow(start=a)
    result = flow.connect(a, b)
    assert result is flow
