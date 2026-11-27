"""The active routing config, and hot reload (PLAN.md §4).

``ConfigManager.current`` is an immutable ``Snapshot``: the loaded config, its routing
table and one adapter per enabled provider. A request reads it once and keeps it, so
requests in flight finish on the config they started with.

A reload (file watch, ``SIGHUP``) runs parse → validate → build → swap:

* an invalid file changes nothing: every problem is logged, and
  ``switchboard_config_reloads_total{result="error"}`` is incremented;
* an unchanged file (same hash) is a no-op;
* adapters are reused for providers whose settings and key did not change, so their
  connection pools stay warm. Replaced adapters are closed only after the longest
  request deadline of the old config, so streams still using them can finish.

The watch reads the file (and the pricing file it selects) once per
``config_watch_seconds`` and reloads when their bytes change. A content check, rather
than file-system events or modification times, sees every edit on every file system:
bind mounts that drop inotify events, file systems with one-second timestamps (where
two saves in the same second look like one), and Kubernetes ConfigMaps, which swap a
symlink instead of writing the file. Each change is reloaded once, so a broken file
logs its errors once, not every second.
"""

import asyncio
import hashlib
import os
import signal
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import structlog

from switchboard.observability.metrics import CONFIG_INFO, CONFIG_RELOADS
from switchboard.providers.base import ProviderAdapter
from switchboard.providers.registry import api_key_for, build_adapter
from switchboard.providers.thinking import ThinkingSidecar
from switchboard.reliability.breaker import Breakers
from switchboard.routing import ConfigError, LoadedConfig, load_routing_config
from switchboard.routing.table import RoutingTable

_log = structlog.get_logger("switchboard.reload")
_RETIRE_MARGIN = 5.0


@dataclass(frozen=True, slots=True)
class Snapshot:
    loaded: LoadedConfig
    table: RoutingTable
    adapters: Mapping[str, ProviderAdapter]
    api_keys: Mapping[str, str | None]  # the key each adapter was built with

    @property
    def version(self) -> str:
        return self.loaded.version


class ConfigManager:
    def __init__(
        self,
        loaded: LoadedConfig,
        *,
        path: Path | None,
        sidecar: ThinkingSidecar,
        breakers: Breakers,
        env: Mapping[str, str] | None = None,
    ) -> None:
        """Must be created inside the event loop (it builds connection pools)."""
        self.path = path
        self._sidecar = sidecar
        self._breakers = breakers
        self._env = env
        self._lock = asyncio.Lock()
        self._retired: dict[ProviderAdapter, asyncio.TimerHandle] = {}
        self._closing: set[asyncio.Task[None]] = set()
        self._background: set[asyncio.Task[str]] = set()
        self._watcher: asyncio.Task[None] | None = None
        self._signal_installed = False
        self.current = self._build(loaded, None)
        _publish_version(loaded.version)

    # --- triggers ----------------------------------------------------------------------

    def start(self, *, watch_seconds: float) -> None:
        """Start the file watch (unless ``watch_seconds`` is 0) and the SIGHUP handler.
        Both are no-ops without a config file."""
        if self.path is None:
            return
        if watch_seconds > 0:
            self._watcher = asyncio.create_task(self._watch(self.path, watch_seconds))
        if hasattr(signal, "SIGHUP"):
            try:
                asyncio.get_running_loop().add_signal_handler(signal.SIGHUP, self.trigger, "signal")
                self._signal_installed = True
            except (NotImplementedError, RuntimeError, ValueError) as exc:  # not main thread
                _log.warning("config.signal_unavailable", detail=str(exc))

    def trigger(self, trigger: str) -> None:
        """Schedule a reload; safe to call from a signal handler."""
        task = asyncio.get_running_loop().create_task(self.reload(trigger))
        self._background.add(task)
        task.add_done_callback(self._background.discard)

    async def reload(self, trigger: str) -> str:
        """Reload from the file: ``success``, ``unchanged`` or ``error``."""
        async with self._lock:
            result = await self._reload(trigger)
        CONFIG_RELOADS.labels(trigger, result).inc()
        return result

    async def _reload(self, trigger: str) -> str:
        if self.path is None:
            _log.warning("config.reload_skipped", trigger=trigger, detail="no config file")
            return "error"
        environ = os.environ if self._env is None else self._env
        try:
            loaded = await asyncio.to_thread(load_routing_config, self.path, env=environ)
        except ConfigError as exc:
            _log.error(
                "config.reload_failed",
                trigger=trigger,
                source=exc.source,
                kept_version=self.current.version,
                problems=[str(issue) for issue in exc.issues],
            )
            return "error"
        if loaded.version == self.current.version:
            return "unchanged"
        old = self.current
        self.current = new = self._build(loaded, old)
        self._retire(old, new)
        _publish_version(loaded.version)
        for warning in loaded.warnings:
            _log.warning("config.warning", problem=str(warning))
        _log.info(
            "config.reloaded",
            trigger=trigger,
            version=loaded.version,
            previous_version=old.version,
            providers_added=sorted(set(new.adapters) - set(old.adapters)),
            providers_removed=sorted(set(old.adapters) - set(new.adapters)),
            providers_rebuilt=sorted(
                n
                for n in set(old.adapters) & set(new.adapters)
                if old.adapters[n] is not new.adapters[n]
            ),
        )
        return "success"

    async def _watch(self, path: Path, interval: float) -> None:
        """Reload when the routing file or the pricing file it selects changes."""
        seen = await self._digest(path)
        while True:
            await asyncio.sleep(interval)
            digest = await self._digest(path)
            if digest is None or digest == seen:
                continue  # unchanged, or briefly missing while an editor replaces it
            seen = digest
            await self.reload("watch")

    async def _digest(self, path: Path) -> bytes | None:
        pricing = self.current.loaded.pricing_path
        parts = [await _digest(p) for p in (path, pricing) if p is not None]
        if any(part is None for part in parts):
            return None
        return hashlib.sha256(b"".join(p for p in parts if p is not None)).digest()

    # --- building and retiring ---------------------------------------------------------

    def _build(self, loaded: LoadedConfig, previous: Snapshot | None) -> Snapshot:
        environ = os.environ if self._env is None else self._env
        adapters: dict[str, ProviderAdapter] = {}
        keys: dict[str, str | None] = {}
        for name, provider in loaded.config.providers.items():
            if name in loaded.disabled_providers:
                continue
            key = api_key_for(provider, environ)
            if (
                previous is not None
                and name in previous.adapters
                and previous.loaded.config.providers.get(name) == provider
                and previous.api_keys.get(name) == key
            ):
                adapters[name] = previous.adapters[name]
            else:
                adapters[name] = build_adapter(name, provider, api_key=key, sidecar=self._sidecar)
            keys[name] = key
        self._breakers.track(loaded.config.models)
        return Snapshot(loaded, RoutingTable(loaded), adapters, keys)

    def _retire(self, old: Snapshot, new: Snapshot) -> None:
        kept = {id(adapter) for adapter in new.adapters.values()}
        grace = _RETIRE_MARGIN + old.table.longest_deadline()
        loop = asyncio.get_running_loop()
        for adapter in old.adapters.values():
            if id(adapter) not in kept and adapter not in self._retired:
                self._retired[adapter] = loop.call_later(grace, self._close_later, adapter)

    def _close_later(self, adapter: ProviderAdapter) -> None:
        self._retired.pop(adapter, None)
        task = asyncio.get_running_loop().create_task(adapter.aclose())
        self._closing.add(task)
        task.add_done_callback(self._closing.discard)

    async def aclose(self) -> None:
        """Stop watching and close every adapter, current and retired."""
        if self._signal_installed:
            asyncio.get_running_loop().remove_signal_handler(signal.SIGHUP)
            self._signal_installed = False
        if self._watcher is not None:
            self._watcher.cancel()
            await asyncio.wait({self._watcher})
        for task in list(self._background):
            task.cancel()
        retired = list(self._retired)
        for handle in self._retired.values():
            handle.cancel()
        self._retired.clear()
        adapters: list[ProviderAdapter] = [*self.current.adapters.values(), *retired]
        await asyncio.gather(*(adapter.aclose() for adapter in adapters))
        if self._closing:
            await asyncio.wait(set(self._closing))


async def _digest(path: Path) -> bytes | None:
    try:
        data = await asyncio.to_thread(path.read_bytes)
    except OSError:
        return None
    return hashlib.sha256(data).digest()


def _publish_version(version: str) -> None:
    CONFIG_INFO.clear()
    CONFIG_INFO.labels(version=version).set(1)
