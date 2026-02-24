import logging
import tiktoken

logger = logging.getLogger(__name__)

_encoder_cache = {}


def _get_encoder(model: str = "gpt-3.5-turbo"):
    """Get or cache a tiktoken encoder for the given model."""
    if model not in _encoder_cache:
        try:
            _encoder_cache[model] = tiktoken.encoding_for_model(model)
        except KeyError:
            # Fallback to cl100k_base for unknown models (covers most modern LLMs)
            _encoder_cache[model] = tiktoken.get_encoding("cl100k_base")
    return _encoder_cache[model]


def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    """Return exact token count for the given text."""
    if not text:
        return 0
    encoder = _get_encoder(model)
    return len(encoder.encode(text))


def truncate_to_tokens(text: str, max_tokens: int, model: str = "gpt-3.5-turbo") -> str:
    """Truncate text to fit within max_tokens."""
    if not text:
        return text
    encoder = _get_encoder(model)
    tokens = encoder.encode(text)
    if len(tokens) <= max_tokens:
        return text
    logger.info(f"Truncating from {len(tokens)} to {max_tokens} tokens")
    return encoder.decode(tokens[:max_tokens])


def trim_messages_to_fit(messages: list[dict], max_tokens: int, model: str = "gpt-3.5-turbo") -> list[dict]:
    """Trim oldest messages first to fit within token budget.

    Always keeps the most recent messages. Returns a new list.
    """
    if not messages:
        return messages

    total = sum(count_tokens(m.get("content", ""), model) for m in messages)
    logger.info(f"Context tokens: {total} (limit: {max_tokens})")

    if total <= max_tokens:
        return messages

    # Drop oldest messages until we fit
    trimmed = list(messages)
    while len(trimmed) > 1:
        dropped = trimmed.pop(0)
        total = sum(count_tokens(m.get("content", ""), model) for m in trimmed)
        logger.info(f"Dropped oldest message ({count_tokens(dropped.get('content', ''), model)} tokens), remaining: {total}")
        if total <= max_tokens:
            break

    return trimmed
