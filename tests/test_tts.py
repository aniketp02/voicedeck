"""
Unit tests for TTS service — ElevenLabs integration mocked.
"""
import asyncio
import pytest
from unittest.mock import patch, MagicMock


async def _collect(async_gen):
    """Helper: collect all items from an async generator."""
    items = []
    async for item in async_gen:
        items.append(item)
    return items


class TestSynthesizeStream:
    @pytest.mark.asyncio
    async def test_yields_chunks_from_async_iterator(self):
        from app.services.tts import synthesize_stream

        chunks = [b"chunk1", b"chunk2", b"chunk3"]

        async def fake_async_iter():
            for c in chunks:
                yield c

        mock_stream = fake_async_iter()
        interrupt = asyncio.Event()

        with patch("app.services.tts._open_audio_stream", return_value=mock_stream):
            result = await _collect(synthesize_stream("Hello world", interrupt))

        assert result == chunks

    @pytest.mark.asyncio
    async def test_stops_on_interrupt_event(self):
        from app.services.tts import synthesize_stream

        interrupt = asyncio.Event()

        async def fake_async_iter():
            yield b"chunk1"
            interrupt.set()  # set interrupt after first chunk
            yield b"chunk2"  # should NOT be yielded
            yield b"chunk3"

        mock_stream = fake_async_iter()

        with patch("app.services.tts._open_audio_stream", return_value=mock_stream):
            result = await _collect(synthesize_stream("Hello world", interrupt))

        # chunk1 yielded, then interrupt checked before chunk2 → stops
        assert result == [b"chunk1"]

    @pytest.mark.asyncio
    async def test_skips_empty_text(self):
        from app.services.tts import synthesize_stream

        interrupt = asyncio.Event()
        with patch("app.services.tts._open_audio_stream") as mock_open:
            result = await _collect(synthesize_stream("   ", interrupt))

        mock_open.assert_not_called()
        assert result == []

    @pytest.mark.asyncio
    async def test_skips_non_bytes_chunks(self):
        from app.services.tts import synthesize_stream

        interrupt = asyncio.Event()

        async def fake_async_iter():
            yield b""            # empty bytes — skip
            yield "not bytes"    # wrong type — skip
            yield b"real_chunk"  # valid

        mock_stream = fake_async_iter()

        with patch("app.services.tts._open_audio_stream", return_value=mock_stream):
            result = await _collect(synthesize_stream("Hello", interrupt))

        assert result == [b"real_chunk"]

    @pytest.mark.asyncio
    async def test_handles_sync_iterator_via_to_thread(self):
        from app.services.tts import synthesize_stream

        chunks = [b"a", b"b", b"c"]
        # A sync iterator (has __iter__ but not __aiter__)
        mock_stream = iter(chunks)

        interrupt = asyncio.Event()

        with patch("app.services.tts._open_audio_stream", return_value=mock_stream):
            result = await _collect(synthesize_stream("Hello", interrupt))

        assert result == chunks
