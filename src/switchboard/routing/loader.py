"""Load and validate ``routing.yaml``, reporting every problem with its YAML location.

Validation runs in three passes, and all problems found are reported together:

1. YAML composition: syntax errors and duplicate keys (which PyYAML would otherwise
   silently overwrite).
2. Schema validation with Pydantic: types, ranges and unknown fields, with "did you
   mean" hints.
3. Semantic checks: every alias target and fallback exists, each model's provider
   exists, required environment variables are set.

The result is an immutable ``LoadedConfig`` whose ``version`` is a short hash of the
file, exposed in ``/ready`` and logs so you can tell which config a process runs.
"""

import difflib
import hashlib
import os
import re
import types
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal, Union, get_args, get_origin

import yaml
from pydantic import BaseModel, ValidationError

from switchboard.routing.config import AliasConfig, ModelKind, RoutingConfig

ConfigPath = tuple[str | int, ...]
Severity = Literal["error", "warning"]

_PLAIN_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]*$")
_MODEL_KEY_RE = re.compile(
    r"^(?P<provider>[a-z0-9][a-z0-9._-]*)/(?P<name>[A-Za-z0-9][A-Za-z0-9._:-]*)$"
)


def format_path(path: ConfigPath) -> str:
    """Render ``("aliases", "smart", "targets", 0)`` as ``aliases.smart.targets[0]``."""
    out = ""
    for part in path:
        if isinstance(part, int):
            out += f"[{part}]"
        elif _PLAIN_KEY_RE.match(part):
            out += f".{part}" if out else part
        else:
            out += f'["{part}"]'
    return out


@dataclass(frozen=True, slots=True)
class ConfigIssue:
    path: ConfigPath
    message: str
    line: int | None = None
    severity: Severity = "error"

    def __str__(self) -> str:
        where = format_path(self.path) or "<root>"
        if self.line is not None:
            where = f"line {self.line}, {where}"
        return f"{where}: {self.message}"


class ConfigError(Exception):
    """The routing config is unusable. ``issues`` lists every problem found."""

    def __init__(self, source: str, issues: Sequence[ConfigIssue]) -> None:
        self.source = source
        self.issues = tuple(sorted(issues, key=lambda i: (i.line or 0, format_path(i.path))))
        super().__init__(self.render())

    def render(self) -> str:
        count = len(self.issues)
        header = f"Invalid routing config {self.source} ({count} problem{'s' * (count != 1)}):"
        return "\n".join([header, *(f"  - {issue}" for issue in self.issues)])


@dataclass(frozen=True, slots=True)
class LoadedConfig:
    config: RoutingConfig
    source: str
    version: str
    warnings: tuple[ConfigIssue, ...] = ()
    disabled_providers: frozenset[str] = frozenset()


def load_routing_config(path: Path, *, env: Mapping[str, str] | None = None) -> LoadedConfig:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise ConfigError(str(path), [ConfigIssue((), "file not found")]) from None
    except OSError as exc:
        raise ConfigError(str(path), [ConfigIssue((), f"cannot read file: {exc}")]) from None
    return parse_routing_config(text, source=str(path), env=env)


def parse_routing_config(
    text: str, *, source: str = "<string>", env: Mapping[str, str] | None = None
) -> LoadedConfig:
    environ: Mapping[str, str] = os.environ if env is None else env
    data, lines, issues = _compose(text, source)
    if data is None:
        raise ConfigError(source, [ConfigIssue((), "the file is empty")])
    if not isinstance(data, dict):
        raise ConfigError(source, [ConfigIssue((), "the top level must be a mapping", 1)])

    try:
        config = RoutingConfig.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(source, [*issues, *_from_validation_error(exc, lines)]) from None

    errors, warnings, disabled = _check_semantics(config, environ, lines)
    issues.extend(errors)
    if issues:
        raise ConfigError(source, issues)
    return LoadedConfig(
        config=config,
        source=source,
        version=hashlib.sha256(text.encode()).hexdigest()[:12],
        warnings=tuple(warnings),
        disabled_providers=frozenset(disabled),
    )


# --- pass 1: YAML -> plain data, plus a map of path -> line -----------------------------


def _compose(text: str, source: str) -> tuple[Any, dict[ConfigPath, int], list[ConfigIssue]]:
    loader = yaml.SafeLoader(text)
    lines: dict[ConfigPath, int] = {}
    issues: list[ConfigIssue] = []
    try:
        root = loader.get_single_node()
        if root is None:
            return None, lines, issues
        active: set[int] = set()

        def build(node: yaml.Node, path: ConfigPath) -> Any:
            lines.setdefault(path, node.start_mark.line + 1)
            if id(node) in active:
                raise ConfigError(source, [ConfigIssue(path, "recursive YAML alias", lines[path])])
            if isinstance(node, yaml.MappingNode):
                active.add(id(node))
                # Keys written in this mapping; "<<" merge keys pull in others, which the
                # mapping's own keys may override. Only repeated explicit keys are errors.
                explicit = {id(k) for k, _ in node.value if k.tag != "tag:yaml.org,2002:merge"}
                loader.flatten_mapping(node)  # resolve "<<" merge keys
                out: dict[str | int, Any] = {}
                explicit_seen: set[str | int] = set()
                for key_node, value_node in node.value:
                    key = loader.construct_object(key_node, deep=True)
                    key_line = key_node.start_mark.line + 1
                    if not isinstance(key, str | int) or isinstance(key, bool):
                        issues.append(
                            ConfigIssue(path, f"keys must be strings, got {key!r}", key_line)
                        )
                        continue
                    child = (*path, key)
                    if id(key_node) in explicit:
                        if key in explicit_seen:
                            issues.append(ConfigIssue(child, f"duplicate key '{key}'", key_line))
                            continue
                        explicit_seen.add(key)
                    lines[child] = key_line
                    out[key] = build(value_node, child)
                active.discard(id(node))
                return out
            if isinstance(node, yaml.SequenceNode):
                active.add(id(node))
                items = [build(item, (*path, i)) for i, item in enumerate(node.value)]
                active.discard(id(node))
                return items
            return loader.construct_object(node, deep=True)

        return build(root, ()), lines, issues
    except yaml.MarkedYAMLError as exc:
        mark = exc.problem_mark or exc.context_mark
        problem = exc.problem or "invalid YAML"
        line = mark.line + 1 if mark else None
        detail = f", column {mark.column + 1}" if mark else ""
        raise ConfigError(
            source, [ConfigIssue((), f"YAML syntax error{detail}: {problem}", line)]
        ) from None
    except yaml.YAMLError as exc:
        raise ConfigError(source, [ConfigIssue((), f"YAML error: {exc}")]) from None
    finally:
        loader.dispose()


def _line_for(path: ConfigPath, lines: Mapping[ConfigPath, int]) -> int | None:
    for end in range(len(path), -1, -1):
        line = lines.get(path[:end])
        if line is not None:
            return line
    return None


# --- pass 2: Pydantic errors -> readable issues -----------------------------------------


def _from_validation_error(
    exc: ValidationError, lines: Mapping[ConfigPath, int]
) -> list[ConfigIssue]:
    issues: list[ConfigIssue] = []
    for err in exc.errors(include_url=False):
        loc: ConfigPath = tuple(err["loc"])
        kind = err["type"]
        if "[key]" in loc:  # a mapping key failed validation (e.g. an invalid alias name)
            cut = loc.index("[key]")
            path = loc[:cut]
            message = f"invalid name: {_clean(err['msg'])}"
        elif kind == "extra_forbidden":
            path = loc
            message = f"unknown field '{loc[-1]}'"
            allowed = _fields_at(loc[:-1])
            hint = _suggest(str(loc[-1]), allowed)
            if hint:
                message += _did_you_mean(hint)
            elif allowed:
                message += f"; expected one of: {', '.join(sorted(allowed))}"
        elif kind == "missing":
            path = loc
            message = "required field is missing"
        else:
            path = loc
            message = _clean(err["msg"])
        issues.append(ConfigIssue(path, message, _line_for(path, lines)))
    return issues


def _clean(msg: str) -> str:
    return msg.removeprefix("Value error, ")


def _unwrap(tp: Any) -> Any:
    while True:
        origin = get_origin(tp)
        if origin is Annotated:
            tp = get_args(tp)[0]
        elif origin in (Union, types.UnionType):
            args = [a for a in get_args(tp) if a is not type(None)]
            if len(args) != 1:
                return tp
            tp = args[0]
        else:
            return tp


def _fields_at(path: ConfigPath) -> list[str]:
    """Field names allowed at ``path`` in the schema (empty if it isn't a model)."""
    tp: Any = RoutingConfig
    for part in path:
        tp = _unwrap(tp)
        if isinstance(tp, type) and issubclass(tp, BaseModel):
            field = tp.model_fields.get(str(part))
            if field is None:
                return []
            tp = field.annotation
        elif get_origin(tp) is dict:
            tp = get_args(tp)[1]
        elif get_origin(tp) is list:
            tp = get_args(tp)[0]
        else:
            return []
    tp = _unwrap(tp)
    if isinstance(tp, type) and issubclass(tp, BaseModel):
        return list(tp.model_fields)
    return []


def _suggest(word: str, candidates: Sequence[str]) -> str | None:
    matches = difflib.get_close_matches(word, candidates, n=1, cutoff=0.6)
    return matches[0] if matches else None


# --- pass 3: cross-references and environment -------------------------------------------


def _check_semantics(
    config: RoutingConfig, env: Mapping[str, str], lines: Mapping[ConfigPath, int]
) -> tuple[list[ConfigIssue], list[ConfigIssue], set[str]]:
    checker = _SemanticChecker(config, env, lines)
    checker.check_providers()
    checker.check_model_names()
    for name, alias in config.aliases.items():
        checker.check_alias(name, alias)
    return checker.errors, checker.warnings, checker.disabled


class _SemanticChecker:
    def __init__(
        self, config: RoutingConfig, env: Mapping[str, str], lines: Mapping[ConfigPath, int]
    ) -> None:
        self.config = config
        self.env = env
        self.lines = lines
        self.errors: list[ConfigIssue] = []
        self.warnings: list[ConfigIssue] = []
        self.disabled: set[str] = set()

    def issue(self, path: ConfigPath, message: str, severity: Severity = "error") -> None:
        bucket = self.errors if severity == "error" else self.warnings
        bucket.append(ConfigIssue(path, message, _line_for(path, self.lines), severity))

    def check_providers(self) -> None:
        for name, provider in self.config.providers.items():
            if not provider.api_key_env or self.env.get(provider.api_key_env):
                continue
            path: ConfigPath = ("providers", name, "api_key_env")
            if provider.optional:
                self.disabled.add(name)
                self.issue(path, f"{provider.api_key_env} is not set; provider disabled", "warning")
            else:
                self.issue(
                    path,
                    f"environment variable {provider.api_key_env} is not set "
                    "(set it, or mark the provider 'optional: true')",
                )

    def check_model_names(self) -> None:
        for key in self.config.models:
            match = _MODEL_KEY_RE.match(key)
            if match is None:
                self.issue(("models", key), "model names must look like '<provider>/<model>'")
            elif match["provider"] not in self.config.providers:
                hint = _suggest(match["provider"], list(self.config.providers))
                self.issue(
                    ("models", key),
                    f"unknown provider '{match['provider']}'" + _did_you_mean(hint),
                )

    def check_alias(self, name: str, alias: AliasConfig) -> None:
        base: ConfigPath = ("aliases", name)
        kinds: set[ModelKind] = set()
        resolved_targets = self._check_refs(
            [((*base, "targets", i, "model"), t.model) for i, t in enumerate(alias.targets)],
            "targets",
            kinds,
        )
        self._check_refs(
            [((*base, "fallbacks", j), fb) for j, fb in enumerate(alias.fallbacks)],
            "fallbacks",
            kinds,
        )
        if len(kinds) > 1:
            self.issue(base, "mixes chat and embedding models")
        if resolved_targets is not None and all(
            model.split("/", 1)[0] in self.disabled for model in resolved_targets
        ):
            self.issue(base, "every target's provider is disabled; requests will fail", "warning")

    def _check_refs(
        self, refs: list[tuple[ConfigPath, str]], section: str, kinds: set[ModelKind]
    ) -> list[str] | None:
        """Check model references; return them if all resolved, else None."""
        seen: set[str] = set()
        all_resolved = True
        for path, model in refs:
            if model in seen:
                self.issue(path, f"'{model}' is listed twice in {section}")
                continue
            seen.add(model)
            config = self.config.models.get(model)
            if config is None:
                all_resolved = False
                hint = _suggest(model, list(self.config.models))
                self.issue(path, f"unknown model '{model}'" + _did_you_mean(hint))
                continue
            kinds.add(config.kind)
        return [model for _, model in refs] if all_resolved else None


def _did_you_mean(hint: str | None) -> str:
    return f" (did you mean '{hint}'?)" if hint else ""
