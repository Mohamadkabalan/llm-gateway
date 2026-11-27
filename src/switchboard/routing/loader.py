"""Load and validate ``routing.yaml``, reporting every problem with its YAML location.

Validation runs in three passes, and all problems found are reported together:

1. YAML composition: syntax errors and duplicate keys (which PyYAML would otherwise
   silently overwrite).
2. Schema validation with Pydantic: types, ranges and unknown fields, with "did you
   mean" hints.
3. Semantic checks: every alias target, fallback and rule reference exists, each
   model's provider exists, merged policies are consistent, rules are reachable, and
   required environment variables are set.

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

from switchboard.accounting.pricing import PriceTable, PricingError, load_pricing
from switchboard.routing.config import (
    ModelKind,
    RouteSpec,
    RoutingConfig,
    Rule,
    merge_policy,
    merge_timeouts,
)

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
    version: str  # covers the routing file and its pricing file
    warnings: tuple[ConfigIssue, ...] = ()
    disabled_providers: frozenset[str] = frozenset()
    pricing: PriceTable | None = None
    pricing_path: Path | None = None


def load_routing_config(path: Path, *, env: Mapping[str, str] | None = None) -> LoadedConfig:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise ConfigError(str(path), [ConfigIssue((), "file not found")]) from None
    except OSError as exc:
        raise ConfigError(str(path), [ConfigIssue((), f"cannot read file: {exc}")]) from None
    return parse_routing_config(text, source=str(path), env=env, base_dir=path.parent)


def parse_routing_config(
    text: str,
    *,
    source: str = "<string>",
    env: Mapping[str, str] | None = None,
    base_dir: Path | None = None,
) -> LoadedConfig:
    """``base_dir`` resolves the ``pricing`` path (default: the working directory)."""
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
    prices, pricing_path, pricing_text = None, None, ""
    if config.pricing is not None:
        pricing_path = (base_dir or Path.cwd()) / config.pricing
        try:
            prices, pricing_text = load_pricing(pricing_path)
        except PricingError as exc:
            issues.append(ConfigIssue(("pricing",), str(exc), _line_for(("pricing",), lines)))
        else:
            issues.extend(_check_prices(config, prices, lines))
    if issues:
        raise ConfigError(source, issues)
    digest = hashlib.sha256(text.encode())
    digest.update(pricing_text.encode())
    return LoadedConfig(
        config=config,
        source=source,
        version=digest.hexdigest()[:12],
        warnings=tuple(warnings),
        disabled_providers=frozenset(disabled),
        pricing=prices,
        pricing_path=pricing_path,
    )


def _check_prices(
    config: RoutingConfig, prices: PriceTable, lines: Mapping[ConfigPath, int]
) -> list[ConfigIssue]:
    """With a pricing file, every model must have a price, so no cost is silently 0."""
    return [
        ConfigIssue(
            ("models", name),
            f"no price for '{name}' in pricing file version {prices.version}",
            _line_for(("models", name), lines),
        )
        for name in config.models
        if name not in prices
    ]


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
    checker.check_timeouts()
    for name, alias in config.aliases.items():
        checker.check_route(("aliases", name), alias)
    checker.check_rules()
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
        defaults = self.config.defaults
        for name, provider in self.config.providers.items():
            for field, base, override in (
                ("retry", defaults.retry, provider.retry),
                ("breaker", defaults.breaker, provider.breaker),
            ):
                try:
                    merge_policy(base, override)
                except ValidationError as exc:
                    self.issue(
                        ("providers", name, field),
                        f"effective {field} policy is invalid: {_first_message(exc)}",
                    )
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

    def check_timeouts(self) -> None:
        """Overrides are validated one level at a time by the schema; this checks the
        merged result each model actually runs with."""
        defaults = self.config.defaults.timeouts
        for key, model in self.config.models.items():
            provider = self.config.providers.get(key.partition("/")[0])
            if provider is None:
                continue  # reported by check_model_names
            try:
                merge_timeouts(defaults, provider.timeouts, model.timeouts)
            except ValidationError as exc:
                path: ConfigPath = (
                    ("models", key, "timeouts") if model.timeouts else ("models", key)
                )
                self.issue(path, f"effective timeouts are invalid: {_first_message(exc)}")

    def check_route(self, base: ConfigPath, route: RouteSpec) -> ModelKind | None:
        """Check an alias or an inline rule route; return its kind if it is unambiguous."""
        kinds: set[ModelKind] = set()
        resolved_targets = self._check_refs(
            [((*base, "targets", i, "model"), t.model) for i, t in enumerate(route.targets)],
            "targets",
            kinds,
        )
        resolved_fallbacks = self._check_refs(
            [((*base, "fallbacks", j), fb) for j, fb in enumerate(route.fallbacks)],
            "fallbacks",
            kinds,
        )
        if len(kinds) > 1:
            self.issue(base, "mixes chat and embedding models")
        candidates = [*(resolved_targets or ()), *(resolved_fallbacks or ())]
        if (
            resolved_targets is not None
            and resolved_fallbacks is not None
            and all(model.split("/", 1)[0] in self.disabled for model in candidates)
        ):
            self.issue(
                base, "every candidate's provider is disabled; requests will fail", "warning"
            )
        if route.hedge is not None:
            self._check_hedge(base, route, resolved_targets)
        return next(iter(kinds)) if len(kinds) == 1 else None

    def _check_hedge(
        self, base: ConfigPath, route: RouteSpec, resolved_targets: list[str] | None
    ) -> None:
        assert route.hedge is not None  # noqa: S101 (narrowing for the type checker)
        path: ConfigPath = (*base, "hedge")
        if not route.fallbacks:
            self.issue(path, "hedging needs at least one fallback to hedge to")
        if ModelKind.EMBEDDING in self._kinds(resolved_targets or []):
            self.issue(path, "hedging applies to chat completions only")
        for model in resolved_targets or ():
            first_token = self._first_token(model)
            if first_token is not None and route.hedge.after >= first_token:
                self.issue(
                    path,
                    f"'after' ({route.hedge.after:g}s) is not shorter than the first-token "
                    f"timeout of {model} ({first_token:g}s), so it never hedges",
                    "warning",
                )

    def check_rules(self) -> None:
        seen: dict[str, int] = {}
        catch_all: str | None = None
        for i, rule in enumerate(self.config.rules):
            base: ConfigPath = ("rules", i)
            if rule.name in seen:
                self.issue((*base, "name"), f"rule name '{rule.name}' is used twice")
            seen.setdefault(rule.name, i)
            if catch_all is not None:
                self.issue(
                    base,
                    f"unreachable: rule '{catch_all}' before it matches every request",
                    "warning",
                )
            if rule.when == type(rule.when)():
                catch_all = catch_all or rule.name
            when_kind = self._check_rule_model(base, rule)
            use_kind = self._check_rule_action(base, rule)
            if when_kind is not None and use_kind is not None and when_kind is not use_kind:
                self.issue(
                    (*base, "use"),
                    f"routes {when_kind.value} requests to {use_kind.value} models",
                )

    def _check_rule_model(self, base: ConfigPath, rule: Rule) -> ModelKind | None:
        requested = rule.when.model
        if requested is None:
            return None
        if requested in self.config.aliases:
            return self._route_kind(self.config.aliases[requested])
        if requested in self.config.models:
            return self.config.models[requested].kind
        hint = _suggest(requested, [*self.config.aliases, *self.config.models])
        self.issue(
            (*base, "when", "model"),
            f"no alias or model named '{requested}'" + _did_you_mean(hint),
        )
        return None

    def _check_rule_action(self, base: ConfigPath, rule: Rule) -> ModelKind | None:
        action = rule.use
        if action.alias is not None:
            alias = self.config.aliases.get(action.alias)
            if alias is None:
                hint = _suggest(action.alias, list(self.config.aliases))
                self.issue(
                    (*base, "use", "alias"),
                    f"unknown alias '{action.alias}'" + _did_you_mean(hint),
                )
                return None
            return self._route_kind(alias)
        route = action.route()
        return None if route is None else self.check_route((*base, "use"), route)

    def _route_kind(self, route: RouteSpec) -> ModelKind | None:
        kinds = self._kinds([t.model for t in route.targets])
        return next(iter(kinds)) if len(kinds) == 1 else None

    def _kinds(self, models: Sequence[str]) -> set[ModelKind]:
        return {self.config.models[m].kind for m in models if m in self.config.models}

    def _first_token(self, model: str) -> float | None:
        config = self.config.models.get(model)
        provider = self.config.providers.get(model.partition("/")[0])
        if config is None or provider is None:
            return None
        try:
            merged = merge_timeouts(
                self.config.defaults.timeouts, provider.timeouts, config.timeouts
            )
        except ValidationError:
            return None  # reported by check_timeouts
        return merged.first_token

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


def _first_message(exc: ValidationError) -> str:
    return str(exc.errors()[0]["msg"]).removeprefix("Value error, ")


def _did_you_mean(hint: str | None) -> str:
    return f" (did you mean '{hint}'?)" if hint else ""
