"""
Small shared helpers used by both agents.py and supervisor.py.
"""


def extract_text(content) -> str:
    """
    Normalize a LangChain message's .content into a plain string.

    Usually .content is already a plain string. But some models -- notably
    Gemini's "thinking" models -- return a list of content blocks instead,
    e.g.:
        [{"type": "text", "text": "...", "extras": {"signature": "..."}}]
    where "extras" carries a thought signature (an opaque token the model
    uses to stay consistent across a multi-step tool-calling turn) that has
    nothing to do with the actual reply. This pulls out just the text.
    """
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and "text" in block:
                parts.append(block["text"])
        return "\n".join(parts)

    return str(content)
