"""The two exchange clients share no code and the public one carries nothing that could sign.

AST and text rules on the sources plus import checks in fresh subprocesses (SPEC rule 3: two
separate clients, no shared signing code; CLAUDE.md: credentials only inside ``demo_account.py``).
"""

from __future__ import annotations

import ast
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

import httpx
import pytest

from kronos_agent.exchange import demo_account, public_market

PUBLIC_PATH = Path(public_market.__file__).resolve()
DEMO_PATH = Path(demo_account.__file__).resolve()
PACKAGE_ROOT = PUBLIC_PATH.parents[1]  # src/kronos_agent
LIVE_HOST = "open-api.bingx.com"

PUBLIC_FORBIDDEN_IMPORTS = {
    "hmac",
    "hashlib",
    "base64",
    "secrets",
    "os",
    "dotenv",
    "subprocess",
    "pathlib",
}
PUBLIC_FORBIDDEN_TEXT = ("X-BX-APIKEY", "signature", "secret", "open-api-vst", "environ", "getenv")
DEMO_FORBIDDEN_IMPORTS = {"pathlib", "dotenv", "subprocess"}
DEMO_ALLOWED_ENV_NAMES = {"BINGX_API_KEY", "BINGX_API_SECRET", "BINGX_HOST", "BINGX_BASE_URL"}
ENV_ACCESSORS = {"environ", "environb", "getenv", "getenvb", "putenv", "unsetenv"}
CREDENTIAL_ENV_NAMES = {"BINGX_API_KEY", "BINGX_API_SECRET"}
SCOPES = (
    ast.Module,
    ast.ClassDef,
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.Lambda,
    ast.ListComp,
    ast.SetComp,
    ast.DictComp,
    ast.GeneratorExp,
)
NAME_BINDERS = (
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.ClassDef,
    ast.ExceptHandler,
    ast.MatchAs,
    ast.MatchStar,
)


@pytest.fixture(scope="module")
def public_source() -> str:
    return PUBLIC_PATH.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def demo_source() -> str:
    return DEMO_PATH.read_text(encoding="utf-8")


def imported_roots(tree: ast.AST) -> set[str]:
    """Top-level names of every ``import``/``from ... import`` anywhere in the tree."""
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            roots.add(node.module.split(".")[0])
    return roots


def imported_modules(tree: ast.AST) -> set[str]:
    """Full dotted module names of every import (relative ones keep their leading dots)."""
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            modules.add("." * node.level + (node.module or ""))
    return modules


def attribute_names(tree: ast.AST) -> set[str]:
    return {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}


def plain_names(tree: ast.AST) -> set[str]:
    return {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}


def string_constants(tree: ast.AST) -> set[str]:
    return {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }


def imported_names(tree: ast.AST) -> set[str]:
    """Names taken by every ``from ... import name`` (before any ``as``)."""
    return {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }


def is_environ(node: ast.AST) -> bool:
    return isinstance(node, ast.Attribute) and node.attr == "environ"


def environ_reads(tree: ast.AST) -> tuple[list[ast.expr], int]:
    """Key of every ``x.environ[key]`` load, ``x.environ.get(key, ...)`` and ``key in x.environ``,
    plus the number of ``.environ`` attributes (the two match only if every use is such a read)."""
    keys: list[ast.expr] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Subscript)
            and isinstance(node.ctx, ast.Load)
            and is_environ(node.value)
        ):
            keys.append(node.slice)
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and is_environ(node.func.value)
            and node.args
            and not isinstance(node.args[0], ast.Starred)
        ):
            keys.append(node.args[0])
        elif (
            isinstance(node, ast.Compare)
            and len(node.ops) == 1
            and isinstance(node.ops[0], (ast.In, ast.NotIn))
            and is_environ(node.comparators[0])
        ):
            keys.append(node.left)
    return keys, sum(1 for node in ast.walk(tree) if is_environ(node))


class LiteralResolver:
    """Resolves expressions of one module to the string literals they can evaluate to.

    A small static analysis that fails closed: a name is followed through every binding in the
    innermost Python scope that binds it (module constants, names bound to them, loop and
    comprehension targets over literal tuples); parameters, imports, definitions, unpacking,
    ``+=``, ``:=``, ``global`` and ``with ... as`` are opaque. Simplified: default values and
    decorators are looked up as if they were inside their function.
    """

    def __init__(self, tree: ast.AST) -> None:
        self.parents = {
            child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)
        }
        self.bindings: dict[tuple[ast.AST, str], list[tuple[str, ast.expr] | None]] = {}
        for node in ast.walk(tree):
            parent = self.parents.get(node)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                if isinstance(parent, ast.AnnAssign) and parent.value is None:
                    continue  # a bare annotation binds nothing
                if isinstance(parent, (ast.Assign, ast.AnnAssign)):
                    self.bind(node, node.id, ("value", parent.value))
                elif isinstance(parent, (ast.For, ast.AsyncFor, ast.comprehension)):
                    self.bind(node, node.id, ("items", parent.iter))
                else:  # unpacking, +=, with ... as; := also binds outside a comprehension
                    self.bind(node, node.id, None, everywhere=isinstance(parent, ast.NamedExpr))
            elif isinstance(node, ast.arg):
                self.bind(node, node.arg, None)
            elif isinstance(node, ast.alias):
                self.bind(node, node.asname or node.name.partition(".")[0], None)
            elif isinstance(node, (ast.Global, ast.Nonlocal)):
                for name in node.names:
                    self.bind(node, name, None, everywhere=True)
            elif isinstance(node, NAME_BINDERS) and node.name:
                self.bind(node, node.name, None)
            elif isinstance(node, ast.MatchMapping) and node.rest:
                self.bind(node, node.rest, None)

    def scopes(self, node: ast.AST) -> list[ast.AST]:
        """Scopes a name used at ``node`` is looked up in, innermost first (a class body is
        visible only to code directly inside it)."""
        chain: list[ast.AST] = []
        while node in self.parents:
            node = self.parents[node]
            if isinstance(node, SCOPES) and not (chain and isinstance(node, ast.ClassDef)):
                chain.append(node)
        return chain

    def bind(
        self,
        node: ast.AST,
        name: str,
        how: tuple[str, ast.expr] | None,
        *,
        everywhere: bool = False,
    ) -> None:
        chain = self.scopes(node)
        for scope in chain if everywhere else chain[:1]:
            self.bindings.setdefault((scope, name), []).append(how)

    def lookup(self, name: ast.Name) -> list[tuple[str, ast.expr] | None]:
        """Bindings of ``name`` in the innermost scope that binds it (opaque if none does)."""
        for scope in self.scopes(name):
            if (scope, name.id) in self.bindings:
                return self.bindings[scope, name.id]
        return [None]  # a builtin or an undefined name

    def strings(
        self,
        node: ast.expr,
        *,
        items: bool = False,
        seen: frozenset[tuple[ast.AST, bool]] = frozenset(),
    ) -> set[str]:
        """Strings ``node`` can evaluate to (with ``items``: the strings it yields when iterated);
        AssertionError for anything it cannot follow, so an indirect key never passes."""
        if not items and isinstance(node, ast.Constant) and isinstance(node.value, str):
            return {node.value}
        if items and isinstance(node, (ast.Tuple, ast.List, ast.Set)):
            return set().union(*(self.strings(elt, seen=seen) for elt in node.elts))
        if isinstance(node, ast.Name) and (node, items) not in seen:
            found = self.lookup(node)
            if all(how is not None and not (items and how[0] == "items") for how in found):
                seen = seen | {(node, items)}
                return set().union(
                    *(
                        self.strings(expr, items=items or kind == "items", seen=seen)
                        for kind, expr in found
                    )
                )
        raise AssertionError(
            f"line {node.lineno}: cannot resolve {ast.unparse(node)!r} to string literals"
        )


def check_demo_environment_reads(source: str) -> None:
    """Every environment access in the demo source reads one of the four allowed names through
    string literals the resolver can follow (shared by the real test and its mutants)."""
    tree = ast.parse(source)
    env_constants = {s for s in string_constants(tree) if s.startswith("BINGX_")}
    assert env_constants == DEMO_ALLOWED_ENV_NAMES
    assert "getenv" not in attribute_names(tree)
    # os.environ is the only door: no environb/getenvb/putenv/unsetenv, no `from os import environ`
    # (aliased or not) and no getattr(os, "environ").
    assert (ENV_ACCESSORS - {"environ"}) & attribute_names(tree) == set()
    named = plain_names(tree) | imported_names(tree) | string_constants(tree)
    assert ENV_ACCESSORS & named == set()
    # Every os.environ use is a read (`[key]`, `.get(key, ...)`, `key in`) whose key resolves to
    # string literals, and together the reads cover exactly the allowed names.
    keys, environ_attributes = environ_reads(tree)
    assert environ_attributes == len(keys), "os.environ used other than as a keyed read"
    resolver = LiteralResolver(tree)
    read = set().union(*(resolver.strings(key) for key in keys))
    assert read == DEMO_ALLOWED_ENV_NAMES, sorted(read)


def package_sources() -> dict[str, str]:
    """Source of every module of the package, keyed by its path relative to the package."""
    return {
        path.relative_to(PACKAGE_ROOT).as_posix(): path.read_text(encoding="utf-8")
        for path in sorted(PACKAGE_ROOT.rglob("*.py"))
    }


def credential_name_modules(sources: Mapping[str, str]) -> set[str]:
    """Modules holding a credential variable name as a string (or bytes) literal."""
    holders: set[str] = set()
    for path, source in sources.items():
        for node in ast.walk(ast.parse(source)):
            value = node.value if isinstance(node, ast.Constant) else None
            if isinstance(value, bytes):
                value = value.decode("latin-1")
            if value in CREDENTIAL_ENV_NAMES:
                holders.add(path)
    return holders


def run_isolated(code: str) -> None:
    """Run ``code`` in a fresh interpreter (same venv) so ``sys.modules`` starts clean."""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr


# -- public market client: nothing that could authenticate ----------------------------------------------


def test_public_client_imports_no_signing_or_environment_modules(public_source):
    tree = ast.parse(public_source)
    assert imported_roots(tree) & PUBLIC_FORBIDDEN_IMPORTS == set()
    for module in imported_modules(tree):
        assert "demo_account" not in module, module


def test_public_client_never_touches_the_environment(public_source):
    tree = ast.parse(public_source)
    assert {"environ", "getenv", "putenv"} & attribute_names(tree) == set()
    assert {"environ", "getenv"} & plain_names(tree) == set()


def test_public_client_text_has_no_auth_vocabulary(public_source):
    lowered = public_source.lower()
    for forbidden in PUBLIC_FORBIDDEN_TEXT:
        assert forbidden.lower() not in lowered, forbidden


def test_public_client_has_no_headers_parameter_and_no_signing_helpers():
    import inspect

    assert "headers" not in inspect.signature(public_market.PublicMarketClient).parameters
    for name in dir(public_market):
        assert "sign" not in name.lower() and "hmac" not in name.lower(), name


# -- demo client: no files, no live host, only the four environment names ------------------------------


def test_demo_client_never_mentions_the_live_host(demo_source):
    assert LIVE_HOST not in demo_source


def test_demo_client_never_opens_files(demo_source):
    tree = ast.parse(demo_source)
    assert "open(" not in demo_source
    assert imported_roots(tree) & DEMO_FORBIDDEN_IMPORTS == set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id != "open"
    for module in imported_modules(tree):
        assert "public_market" not in module, module


def test_demo_client_reads_only_the_allowed_environment_names(demo_source):
    check_demo_environment_reads(demo_source)


# Indirect reads the check must refuse, each appended to the real demo source; a check that took
# any name as a key without resolving it would let every one of them through.
INDIRECT_ENV_READS = {
    "module-constant": 'HOME_VAR = "HOME"\n\n\ndef _read():\n    return os.environ[HOME_VAR]\n',
    "parameter": "def _read(var):\n    return os.environ.get(var)\n",
    "rebound-name": (
        "def _read():\n"
        "    name = API_KEY_ENV\n"
        "    name = os.getcwd()\n"
        "    return os.environ.get(name)\n"
    ),
    "loop-over-literal": 'for name in ("HOME",):\n    os.environ.get(name)\n',
    "environ-alias": '_env = os.environ\n_env["HOME"]\n',
    "from-import": 'from os import environ\n\nenviron["HOME"]\n',
    "aliased-from-import": 'from os import environ as env\n\nenv["HOME"]\n',
    "getattr": 'getattr(os, "environ")["HOME"]\n',
    "bytes-environment": 'os.environb[b"HOME"]\n',
    "membership": '"HOME" in os.environ\n',
    "write": 'os.environ[API_KEY_ENV] = "x"\n',
}
# Reads through literals the check must keep accepting, appended the same way.
LITERAL_ENV_READS = {
    "membership-of-a-constant": "def _is_set():\n    return API_KEY_ENV in os.environ\n",
    "constant-of-a-constant": (
        'KEY_NAME = API_KEY_ENV\n\n\ndef _key():\n    return os.environ.get(KEY_NAME, "")\n'
    ),
    "loop-name-bound-elsewhere": "def _unrelated(name):\n    return name.upper()\n",
}


@pytest.mark.parametrize("snippet", list(INDIRECT_ENV_READS.values()), ids=list(INDIRECT_ENV_READS))
def test_environment_name_check_rejects_indirect_reads(demo_source, snippet):
    with pytest.raises(AssertionError):
        check_demo_environment_reads(demo_source + "\n\n" + snippet)


@pytest.mark.parametrize("snippet", list(LITERAL_ENV_READS.values()), ids=list(LITERAL_ENV_READS))
def test_environment_name_check_accepts_reads_through_literals(demo_source, snippet):
    check_demo_environment_reads(demo_source + "\n\n" + snippet)


def test_demo_client_compares_against_the_literal_host_at_least_three_times(demo_source):
    tree = ast.parse(demo_source)
    literal_uses = sum(
        1
        for node in ast.walk(tree)
        if isinstance(node, ast.Compare)
        and any(
            isinstance(c, ast.Constant) and c.value == "open-api-vst.bingx.com"
            for c in node.comparators
        )
    )
    assert literal_uses >= 3


# -- no cross imports; hmac and the credential names only in the demo module -----------------------------


def test_importing_the_public_client_loads_neither_demo_client_nor_hmac():
    run_isolated(
        "import sys\n"
        "import kronos_agent.exchange.public_market\n"
        "assert 'kronos_agent.exchange.demo_account' not in sys.modules, 'demo client imported'\n"
        "assert 'hmac' not in sys.modules, 'hmac imported'\n"
    )


def test_importing_the_demo_client_does_not_load_the_public_client():
    run_isolated(
        "import sys\n"
        "import kronos_agent.exchange.demo_account\n"
        "assert 'kronos_agent.exchange.public_market' not in sys.modules, 'public client imported'\n"
    )


def test_hmac_is_imported_only_in_demo_account():
    importers: set[str] = set()
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        if "hmac" in imported_roots(tree):
            importers.add(path.relative_to(PACKAGE_ROOT).as_posix())
    assert importers == {"exchange/demo_account.py"}


def test_bingx_host_literals_live_only_in_the_exchange_package():
    # every request to BingX goes through one of the two pinned clients; the package docstring
    # of exchange/ names both hosts
    holders = {
        path
        for path, source in package_sources().items()
        if any("bingx.com" in text for text in string_constants(ast.parse(source)))
    }
    assert holders <= {
        "exchange/__init__.py",
        "exchange/public_market.py",
        "exchange/demo_account.py",
    }, sorted(holders)
    assert {"exchange/public_market.py", "exchange/demo_account.py"} <= holders


def test_credential_env_names_appear_only_in_demo_account():
    # CLAUDE.md: credentials are read only inside demo_account.py. A help text elsewhere may
    # mention the names inside a longer string or import demo_account's constants.
    assert credential_name_modules(package_sources()) == {"exchange/demo_account.py"}


@pytest.mark.parametrize(
    "snippet",
    [
        'API_KEY_NAME = "BINGX_API_KEY"',
        'os.environ.get("BINGX_API_SECRET")',
        'os.environb[b"BINGX_API_SECRET"]',
    ],
)
def test_credential_name_check_flags_a_name_in_another_module(snippet):
    sources = package_sources()
    sources["exchange/public_market.py"] += "\n" + snippet + "\n"
    assert credential_name_modules(sources) == {
        "exchange/demo_account.py",
        "exchange/public_market.py",
    }


def test_exchange_package_init_imports_nothing():
    init = ast.parse((PACKAGE_ROOT / "exchange" / "__init__.py").read_text(encoding="utf-8"))
    assert imported_modules(init) == set()


# -- distinct private transports ------------------------------------------------------------------------------


def test_transport_classes_are_distinct_private_and_not_shared():
    public_transport = public_market._PublicPinnedTransport
    demo_transport = demo_account._DemoPinnedTransport
    assert public_transport is not demo_transport
    assert public_transport.__name__.startswith("_") and demo_transport.__name__.startswith("_")
    assert issubclass(public_transport, httpx.BaseTransport) and issubclass(
        demo_transport, httpx.BaseTransport
    )
    assert not issubclass(public_transport, demo_transport) and not issubclass(
        demo_transport, public_transport
    )
    assert public_transport.__module__ == public_market.__name__
    assert demo_transport.__module__ == demo_account.__name__
    shared = set(public_transport.__mro__) & set(demo_transport.__mro__)
    assert shared == {httpx.BaseTransport, object}
    assert public_transport.handle_request is not demo_transport.handle_request


def test_exception_types_are_module_specific():
    assert public_market.HostNotAllowedError is not demo_account.HostNotAllowedError
    assert public_market.UnexpectedRedirect is not demo_account.UnexpectedRedirect
    assert not issubclass(public_market.HostNotAllowedError, demo_account.DemoAccountError)
    assert not issubclass(demo_account.HostNotAllowedError, public_market.PublicMarketError)
