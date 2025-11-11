# AI INSTRUCTION:
# Provide reusable prompt fragments and templates specific to search.
# These can be used by the generator to form the final system message.

BASE_GUARDRAILS = """\
You must answer using only the provided context.
If the context does not contain the answer, say:
"I don't know based on available information."
Always cite which snippets you used (by their source or id).
"""

def build_system_prompt(persona_name: str, style: str, directives: str) -> str:
    return f"""You are {persona_name}.
Your style: {style}

Directives:
{directives}

{BASE_GUARDRAILS}
"""
