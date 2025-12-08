import re

from model_loader import llm


META_PREFIX_RE = re.compile(
    r"""^\s*
        (?:
          (?:[Tt]he\s+)?(?:speaker|narrator|host|video|segment|scene|clip)\s+
          (?:says|says\s+that|mentions|talks\s+about|discusses|describes|announces|states|explains)\s*:?\s*
        )
    """,
    re.X,
)


def clean_meta(text: str) -> str:
    text = text.strip()
    text = META_PREFIX_RE.sub("", text)
    text = re.sub(r"^(```|Output\s*:|Answer\s*:)\s*", "", text, flags=re.I).strip()
    text = text.strip().strip('“”"\'')
    text = text.splitlines()[0].strip()
    return text or "None"


def extract_speech_from_caption_with_llm(caption_text: str, speech_summary: str, start: str, end: str, duration: str) -> str:
    """
    LLaMA를 이용해 특정 시간대 caption + 전체 speech summary를 기반으로
    그 시간대의 speech 내용을 추출
    """

    system_prompt = f"""
    You are a structured information extraction engine that extracts spoken speech at a given video time segment.
    RULES:
    - Output ONLY the speech text as one plain line. No labels, no quotes, no JSON, no code fences, no preambles (e.g., "The speaker ..."), no explanations.
    - Use only content present in the given speech summary; do NOT invent.
    - Select the portion most relevant to the segment {start}-{end} and its caption. Total Range is (00.00 - {duration}).
    - If no relevant speech exists, output: None
    - Keep the input language and keep it concise.
    - Preserve inner quotes if present. Output must NOT start with meta phrases.
    """

    user_prompt = f"""
    Now process this input:
    
    Time: {start}-{end}
    Caption: "{caption_text}"
    Speech summary: "{speech_summary}"

    Output (speech only, extracted from the summary):
    """
    output = llm(system_prompt, user_prompt)
    output = clean_meta(output)

    if not output or output.lower() in {"none", "(none)"}:
        return "None"

    return output
