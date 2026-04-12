"""
Unit tests for TTS services — both ElevenLabs and Deepgram providers.
"""
import asyncio
import pytest
from unittest.mock import patch, AsyncMock, MagicMock


async def _collect(async_gen):
    """Helper: collect all items from an async generator."""
    items = []
    async for item in async_gen:
        items.append(item)
    return items


# ---------------------------------------------------------------------------
# ElevenLabs provider (tts_elevenlabs.py)
# ---------------------------------------------------------------------------


class TestElevenLabsSynthesizeStream:
    @pytest.mark.asyncio
    async def test_yields_chunks_from_async_iterator(self):
        from app.services.tts_elevenlabs import synthesize_stream

        chunks = [b"chunk1", b"chunk2", b"chunk3"]

        async def fake_async_iter():
            for c in chunks:
                yield c

        mock_stream = fake_async_iter()
        interrupt = asyncio.Event()

        with patch("app.services.tts_elevenlabs.settings") as mock_settings, patch(
            "app.services.tts_elevenlabs._open_audio_stream", return_value=mock_stream
        ):
            mock_settings.elevenlabs_api_key = "test-key"
            result = await _collect(synthesize_stream("Hello world", interrupt))

        assert result == chunks

    @pytest.mark.asyncio
    async def test_stops_on_interrupt_event(self):
        from app.services.tts_elevenlabs import synthesize_stream

        interrupt = asyncio.Event()

        async def fake_async_iter():
            yield b"chunk1"
            interrupt.set()
            yield b"chunk2"
            yield b"chunk3"

        mock_stream = fake_async_iter()

        with patch("app.services.tts_elevenlabs.settings") as mock_settings, patch(
            "app.services.tts_elevenlabs._open_audio_stream", return_value=mock_stream
        ):
            mock_settings.elevenlabs_api_key = "test-key"
            result = await _collect(synthesize_stream("Hello world", interrupt))

        assert result == [b"chunk1"]

    @pytest.mark.asyncio
    async def test_skips_empty_text(self):
        from app.services.tts_elevenlabs import synthesize_stream

        interrupt = asyncio.Event()
        with patch("app.services.tts_elevenlabs._open_audio_stream") as mock_open, patch(
            "app.services.tts_elevenlabs.settings"
        ) as mock_settings:
            mock_settings.elevenlabs_api_key = "test-key"
            result = await _collect(synthesize_stream("   ", interrupt))

        mock_open.assert_not_called()
        assert result == []

    @pytest.mark.asyncio
    async def test_skips_non_bytes_chunks(self):
        from app.services.tts_elevenlabs import synthesize_stream

        interrupt = asyncio.Event()

        async def fake_async_iter():
            yield b""
            yield "not bytes"
            yield b"real_chunk"

        mock_stream = fake_async_iter()

        with patch("app.services.tts_elevenlabs.settings") as mock_settings, patch(
            "app.services.tts_elevenlabs._open_audio_stream", return_value=mock_stream
        ):
            mock_settings.elevenlabs_api_key = "test-key"
            result = await _collect(synthesize_stream("Hello", interrupt))

        assert result == [b"real_chunk"]

    @pytest.mark.asyncio
    async def test_handles_sync_iterator_via_to_thread(self):
        from app.services.tts_elevenlabs import synthesize_stream

        chunks = [b"a", b"b", b"c"]
        mock_stream = iter(chunks)
        interrupt = asyncio.Event()

        with patch("app.services.tts_elevenlabs.settings") as mock_settings, patch(
            "app.services.tts_elevenlabs._open_audio_stream", return_value=mock_stream
        ):
            mock_settings.elevenlabs_api_key = "test-key"
            result = await _collect(synthesize_stream("Hello", interrupt))

        assert result == chunks

    @pytest.mark.asyncio
    async def test_raises_when_api_key_missing(self):
        from app.services.tts_elevenlabs import synthesize_stream

        interrupt = asyncio.Event()
        with patch("app.services.tts_elevenlabs.settings") as mock_settings:
            mock_settings.elevenlabs_api_key = ""
            with pytest.raises(RuntimeError, match="ELEVENLABS_API_KEY"):
                await _collect(synthesize_stream("Hello", interrupt))


# ---------------------------------------------------------------------------
# Deepgram TTS provider (tts_deepgram.py)
# ---------------------------------------------------------------------------


class TestDeepgramSynthesizeStream:
    @pytest.mark.asyncio
    async def test_yields_chunks_from_http_stream(self):
        from app.services.tts_deepgram import synthesize_stream

        chunks = [b"audio1", b"audio2", b"audio3"]
        interrupt = asyncio.Event()

        mock_response = AsyncMock()
        mock_response.status_code = 200

        async def fake_aiter_bytes(chunk_size=4096):
            for c in chunks:
                yield c

        mock_response.aiter_bytes = fake_aiter_bytes
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        with patch("app.services.tts_deepgram._get_client") as mock_client:
            mock_http = MagicMock()
            mock_http.stream.return_value = mock_response
            mock_client.return_value = mock_http

            result = await _collect(synthesize_stream("Hello world", interrupt))

        assert result == chunks

    @pytest.mark.asyncio
    async def test_skips_empty_text(self):
        from app.services.tts_deepgram import synthesize_stream

        interrupt = asyncio.Event()
        with patch("app.services.tts_deepgram._get_client") as mock_client:
            result = await _collect(synthesize_stream("   ", interrupt))

        mock_client.assert_not_called()
        assert result == []

    @pytest.mark.asyncio
    async def test_stops_on_interrupt(self):
        from app.services.tts_deepgram import synthesize_stream

        interrupt = asyncio.Event()

        mock_response = AsyncMock()
        mock_response.status_code = 200

        async def fake_aiter_bytes(chunk_size=4096):
            yield b"chunk1"
            interrupt.set()
            yield b"chunk2"

        mock_response.aiter_bytes = fake_aiter_bytes
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        with patch("app.services.tts_deepgram._get_client") as mock_client:
            mock_http = MagicMock()
            mock_http.stream.return_value = mock_response
            mock_client.return_value = mock_http

            result = await _collect(synthesize_stream("Hello world", interrupt))

        assert result == [b"chunk1"]

    @pytest.mark.asyncio
    async def test_raises_on_non_200_response(self):
        from app.services.tts_deepgram import synthesize_stream

        interrupt = asyncio.Event()

        mock_response = AsyncMock()
        mock_response.status_code = 401
        mock_response.aread = AsyncMock(return_value=b'{"error": "Unauthorized"}')
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        with patch("app.services.tts_deepgram._get_client") as mock_client:
            mock_http = MagicMock()
            mock_http.stream.return_value = mock_response
            mock_client.return_value = mock_http

            with pytest.raises(RuntimeError, match="HTTP 401"):
                await _collect(synthesize_stream("Hello world", interrupt))
