"""Incremental Server-Sent Events decoder.

Upstream bytes arrive in arbitrary pieces: one network read may hold half an event or
several events. ``SSEDecoder.feed`` buffers incomplete lines and returns only complete
events, so the output does not depend on how the stream was split (a property test
checks this). Line endings may be ``\\n``, ``\\r\\n`` or ``\\r``, per the HTML spec.

Event data stays as ``bytes``: ``orjson.loads`` accepts bytes, so the hot path never
decodes and re-encodes text.
"""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SSEEvent:
    data: bytes
    event: str | None = None


class SSEDecoder:
    def __init__(self) -> None:
        self._pending = b""
        self._data: list[bytes] = []
        self._event: str | None = None

    def feed(self, chunk: bytes) -> list[SSEEvent]:
        if self._pending:
            chunk = self._pending + chunk
            self._pending = b""
        lines = chunk.splitlines(keepends=True)
        if not lines:
            return []
        # The last line is incomplete unless it ends with a line break. A trailing "\r"
        # might be the first half of "\r\n", so it waits for the next chunk too.
        last = lines[-1]
        if last.endswith(b"\r") or not last.endswith(b"\n"):
            self._pending = lines.pop()

        events: list[SSEEvent] = []
        for raw in lines:
            self._line(raw.rstrip(b"\r\n"), events)
        return events

    def close(self) -> list[SSEEvent]:
        """Flush at end of stream.

        A held line that ended in ``\\r`` is complete and is processed. Per the spec, an
        unterminated line and an event without its closing blank line are discarded.
        """
        events: list[SSEEvent] = []
        pending, self._pending = self._pending, b""
        if pending.endswith(b"\r"):
            self._line(pending[:-1], events)
        self._data = []
        self._event = None
        return events

    def _line(self, line: bytes, events: list[SSEEvent]) -> None:
        if line:
            self._field(line)
        else:
            self._dispatch(events)

    def _field(self, line: bytes) -> None:
        if line.startswith(b":"):
            return  # comment, e.g. a keep-alive ": ping"
        name, sep, value = line.partition(b":")
        if sep and value.startswith(b" "):
            value = value[1:]
        if name == b"data":
            self._data.append(value)
        elif name == b"event":
            self._event = value.decode("utf-8", "replace")
        # "id" and "retry" are not used by any provider stream we consume.

    def _dispatch(self, events: list[SSEEvent]) -> None:
        if self._data:
            data = self._data[0] if len(self._data) == 1 else b"\n".join(self._data)
            events.append(SSEEvent(data=data, event=self._event))
        self._data = []
        self._event = None
