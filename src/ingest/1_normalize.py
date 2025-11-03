# AI INSTRUCTION:
# Create a Python module in this file that defines a safe text normalization utility for Markdown or plain text.
# It should:
# - Strip control characters (except newlines/tabs)
# - Collapse redundant whitespace
# - Normalize Unicode (NFKC) and unescape HTML entities
# - Protect and restore fenced code blocks (``` or ~~~)
# - Collapse 3+ newlines into max 2
# Return a `normalize_text(raw: str) -> str` function.
# Include helper functions `_protect_code_blocks` and `_restore_code_blocks`.
# Use `re`, `unicodedata`, and `html.unescape`. 
# Add clear comments explaining each section.


import re
import unicodedata
import html

# -------------------------------------
# Helper: Protect fenced code blocks (``` or ~~~) in Markdown
# Replaces code blocks with placeholders, returning (text, code_blocks) tuple.
def _protect_code_blocks(text):
    # Pattern matches both ``` and ~~~ fenced code blocks 
    pattern = re.compile(
        r"(^|\n)(?P<fence>(`{3,}|~{3,}))[^\n]*\n"    # opening fence (with optional lang)
        r"(?P<code>.*?)(\n(?P=fence)[ \t]*\n?)",     # content and closing fence
        re.DOTALL
    )
    code_blocks = []
    def _replace(m):
        idx = len(code_blocks)
        code_blocks.append(m.group())
        return f"\n@@CODEBLOCK_{idx}@@\n"
    protected = pattern.sub(_replace, text)
    return protected, code_blocks

# -------------------------------------
# Helper: Restore code block placeholders to original code blocks
def _restore_code_blocks(text, code_blocks):
    def _restore(m):
        idx = int(m.group(1))
        return code_blocks[idx]
    restored = re.sub(r"@@CODEBLOCK_(\d+)@@", _restore, text)
    return restored

# -------------------------------------
# Main normalization function
def normalize_text(raw: str) -> str:
    """
    Normalizes input Markdown or plain text.
    - Strips control characters (except newlines/tabs)
    - Collapses redundant whitespace
    - Normalizes Unicode (NFKC)
    - Unescapes HTML entities
    - Protects and restores fenced code blocks
    - Collapses 3+ consecutive newlines into max 2
    """
    # 1. Protect code blocks
    protected, code_blocks = _protect_code_blocks(raw)

    # 2. Strip unwanted control chars (allow \n, \t)
    #   \x00-\x08, \x0b-\x0c, \x0e-\x1f, \x7f
    cleaned = re.sub(r"[^\S\r\n\t]", " ", protected) # replace all remaining whitespace except \n\t with space
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", cleaned)

    # 3. Normalize Unicode (NFKC)
    cleaned = unicodedata.normalize("NFKC", cleaned)

    # 4. Unescape HTML entities
    cleaned = html.unescape(cleaned)

    # 5. Collapse redundant whitespace (spaces/tabs)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)

    # 6. Collapse 3+ consecutive newlines to max 2
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

    # 7. Strip spaces at line boundaries
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    cleaned = re.sub(r"\n[ \t]+", "\n", cleaned)

    # 8. Restore code blocks
    restored = _restore_code_blocks(cleaned, code_blocks)

    # 9. Final trim
    return restored.strip()
