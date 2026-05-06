from __future__ import annotations
from typing import Dict, List



TAXONOMY_CLASSIFY_PROMPT = """Refer Figure 7 in Appendix
"""


AUGMENT_PROMPT = """Refer Figure 8 in Appendix"""


TRANSLATE_PROMPT = """Refer Figure 9 in Appendix
"""

BACK_TRANSLATE_PROMPT = """Refer Figure 9 in Appendix
"""


RESPONSE_PROMPT_SINGLE = """Refer Figure 10 in Appendix
"""

RESPONSE_PROMPT_MULTI = """Refer Figure 10 in Appendix
"""


RUBRIC_PROMPT = """Refer Figure 10 in Appendix
"""


def fewshot_taxonomy_messages(prompt: str) -> List[Dict[str, str]]:
    """Chat-format wrapper for taxonomy classification."""
    return [
        {"role": "system", "content": "You are a precise classifier that outputs only valid JSON."},
        {"role": "user", "content": TAXONOMY_CLASSIFY_PROMPT.format(prompt=prompt)},
    ]
