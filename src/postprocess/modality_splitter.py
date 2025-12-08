import json
import re

from model_loader import llm


def split_modality_caption_with_llm(caption_text: str) -> dict:
    """LLM을 이용해 Visual/Audio로 분리"""
    system_prompt = """
    You output ONLY valid JSON with exactly two keys:
    {"visual": string, "audio": string}

    DEFINITIONS
    - visual: strictly what is seen.
    - audio: ANY audible content explicitly mentioned in the text, including
    speech/dialogue/announcer/narration, crowd reactions (cheer, boo, applause),
    music/singing, environmental/FX (engine, footsteps, wind, beep, whistle, horn), alarms.

    EXTRACTION RULES
    - Find ALL sound mentions; do not drop any. Preserve the order in the input.
    - Convert each sound mention to a concise phrase (noun/verb). Keep quotes for speech if present.
    - Join multiple audio mentions with ", " (comma+space).
    - Do NOT invent unheard details. If none, set "audio" to "".
    - Keep the same language as the input.
    - JSON only. No extra text, no code fences, no trailing commas.
    """

    user_prompt = f"""
        Example:
        Input: "A woman plays the piano while singing 'I love you'. Applause erupts and a bell rings."
        Output:
        {{"visual":"A woman plays the piano.","audio":"Singing 'I love you', applause, bell rings."}}
        ---
        Now process the segment below.

        Input: {caption_text}
        Output JSON:
        """
    text = llm(system_prompt, user_prompt)

    BAD_ESCAPE_RE = re.compile(r'(?<!\\)\\(?!["\\/bfnrtu])')

    def _sanitize_invalid_escapes(raw: str) -> str:
        return BAD_ESCAPE_RE.sub(lambda m: "\\\\" + m.group(0)[1:], raw)

    def try_parse_json(s: str):
        i = s.find("{")
        if i != -1:
            s = s[i:]
        try:
            return json.loads(s)
        except json.JSONDecodeError as e:
            if "Invalid \\escape" not in str(e):
                raise
            cleaned = _sanitize_invalid_escapes(s)
            return json.loads(cleaned)

    try:
        obj = try_parse_json(text)
    except Exception:
        repair_prompt = (
            'The previous output was not valid JSON with keys {"visual","audio"}.\\n'
            "Fix it to valid JSON now. Do not add extra keys.\\n"
            f"Previous:\\n{text}\\n"
            "Output JSON only:"
        )
        text2 = llm(system_prompt, repair_prompt)
        obj = try_parse_json(text2)

    visual = obj.get("visual", "")
    audio = obj.get("audio", "")
    if not isinstance(visual, str):
        visual = str(visual)
    if not isinstance(audio, str):
        audio = str(audio)

    speech_markers = [r"\\bsaid\\b", r"\\bsays\\b", r"\\bquote\\b", r"\\\".+?\\\""]
    if any(re.search(p, audio, flags=re.IGNORECASE) for p in speech_markers):
        audio = ""

    return {"visual": visual.strip(), "audio": audio.strip()}
