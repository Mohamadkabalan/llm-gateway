from hypothesis import given
from hypothesis import strategies as st

from switchboard.providers.sse import SSEDecoder, SSEEvent

OPENAI_STREAM = (
    b'data: {"id":"1","choices":[{"delta":{"role":"assistant","content":""}}]}\n\n'
    b'data: {"id":"1","choices":[{"delta":{"content":"Hel"}}]}\n\n'
    b": keep-alive comment\n\n"
    b'data: {"id":"1","choices":[{"delta":{"content":"lo \xe2\x9c\x93"}}]}\n\n'
    b"event: error\ndata: line one\ndata: line two\n\n"
    b"data: [DONE]\n\n"
)
EXPECTED = [
    SSEEvent(b'{"id":"1","choices":[{"delta":{"role":"assistant","content":""}}]}'),
    SSEEvent(b'{"id":"1","choices":[{"delta":{"content":"Hel"}}]}'),
    SSEEvent(b'{"id":"1","choices":[{"delta":{"content":"lo \xe2\x9c\x93"}}]}'),
    SSEEvent(b"line one\nline two", event="error"),
    SSEEvent(b"[DONE]"),
]


def decode_all(parts: list[bytes]) -> list[SSEEvent]:
    decoder = SSEDecoder()
    events = [event for part in parts for event in decoder.feed(part)]
    return events + decoder.close()


def test_decodes_a_whole_stream() -> None:
    assert decode_all([OPENAI_STREAM]) == EXPECTED


def test_comments_and_events_without_data_produce_nothing() -> None:
    assert decode_all([b": ping\n\nevent: noop\n\nid: 7\nretry: 100\n\n"]) == []


def test_optional_space_after_colon() -> None:
    assert decode_all([b"data:x\n\ndata:  y\n\n"]) == [SSEEvent(b"x"), SSEEvent(b" y")]


def test_incomplete_final_event_is_discarded() -> None:
    assert decode_all([b"data: complete\n\ndata: cut off"]) == [SSEEvent(b"complete")]


def test_cr_line_endings_held_across_chunks() -> None:
    # "\r" at the end of a chunk may be half of "\r\n"; it must not end the event early.
    assert decode_all([b"data: a\r", b"\ndata: b\r\n\r\n"]) == [SSEEvent(b"a\nb")]


def test_lone_cr_line_endings() -> None:
    assert decode_all([b"data: a\r\rdata: b\r\r"]) == [SSEEvent(b"a"), SSEEvent(b"b")]


@given(st.data())
def test_any_split_of_the_stream_gives_the_same_events(data: st.DataObject) -> None:
    for stream in (OPENAI_STREAM, OPENAI_STREAM.replace(b"\n", b"\r\n")):
        cuts = sorted(
            data.draw(st.lists(st.integers(min_value=0, max_value=len(stream)), max_size=20))
        )
        parts = [stream[a:b] for a, b in zip([0, *cuts], [*cuts, len(stream)], strict=True)]
        assert decode_all(parts) == EXPECTED
