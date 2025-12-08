import json

from model_loader import llm
from schema_utils import (
    SCHEMA,
    ValidationError,
    _fix_attributes_in_json,
    _normalize_object_attributes,
    validate,
)


def extract_info_with_llm(video_id, seg_idx, start, end, text, not_json_dir):
    system_prompt = """
    You are a structured information extraction engine that outputs ONLY valid JSON for an MPEG-7–style schema.
    Follow ALL rules strictly:

    GENERAL
    - Output JSON ONLY. No code fences, no explanations, no trailing text.
    - Use double quotes for all keys/strings. No comments. No trailing commas.
    - If a field is unknown, use "" (empty string) or [] (empty array). Do NOT invent facts.
    - Keep key order exactly as the schema lists. Do not add extra keys.
    - Escape any double quotes inside string values (use \\"), and avoid raw newlines inside strings unless necessary.

    ID & REFERENTIAL INTEGRITY
    - "event_id" must be "E_<video_id>_<start>_<end>".
    - Sanitize <video_id> by replacing non-alphanumeric chars with "_" (keep case).
    - "objects[].object_id" must be unique like "O001", "O002", … (3 digits).
    - "objects[].attributes" MUST be a flat JSON object of string values only.
    - "actors[].actor_id" must be unique like "A001", "A002", … (3 digits).
    - "actors[].ref_object" must reference an existing objects[].object_id.
    - "event.actors" and "event.objects" are arrays of IDs that must exist above.

    TYPES & ENUMS
    - "time.start" and "time.end" are strings echoing the given inputs (no reformat).
    - "LOD.abstract_topic" is an array of strings; "scene_topic", "summary", "implications" are strings.

    SCHEMA (required keys in this exact order)
    {
        "video_id": string,
        "event_id": string,
        "tags": string[],
        "objects": [
        {
        "object_id": string,
        "name": string,
        "attributes": { string: string }
        }
        ],
        "actors": [
            {
            "actor_id": string,
            "name": string,
            "ref_object": string,
            "role": string,
            "entity": string
            }
        ],
        "event": {
            "event_id": string,
            "name": string,
            "time": { "start": string, "end": string },
            "actors": string[],
            "objects": string[]
        },
        "LOD": {
            "abstract_topic": string[],
            "scene_topic": string,
            "summary": string,
            "implications": string
        }
    }
    
    FIELD GUIDELINES (minimal)
    - "tags": 3–6 topic keywords in snake_case (nouns only; no punctuation).
    e.g., ["economy", "central_bank", "interest_rate"]
    - "LOD.abstract_topic": very coarse category label (keep input language).
    e.g., "economy news", "non-violent scene"
    - "LOD.scene_topic": one-sentence event summary (include actor/action; keep input language).
    e.g., "The Bank of Korea governor announces that the base rate is kept on hold."
    - "LOD.implications": short phrase on significance/impact.
    e.g., "Major economics,finance news event"
    - "objects[].attributes.*" values must be concise (<=2 sentences / ~200 chars) and descriptive of the object. Apply the same limit to "LOD.scene_topic" and "LOD.implications".
    """
    user_prompt = f"""

    Now process the following input:

    Input segment:
    Video {video_id}, time {start}-{end}, description: {text}

    Output JSON:
    """

    response = llm(system_prompt, user_prompt)
    schema_text = json.dumps(SCHEMA, ensure_ascii=False, separators=(",", ":"))

    try:
        response = _fix_attributes_in_json(response)
        obj = json.loads(response)
        _normalize_object_attributes(obj)
        validate(obj, SCHEMA)
        return obj
    except (json.JSONDecodeError, ValidationError):
        repair_user = (
            "The previous output was invalid JSON or did not match the schema. "
            "Fix it to match the JSON Schema EXACTLY. Return JSON ONLY.\n\n"
            f"SCHEMA:\n{schema_text}\n\n"
            f"ORIGINAL:\n{response}\n"
        )
        text2 = llm(system_prompt, repair_user)
        try:
            text2 = _fix_attributes_in_json(text2)
            obj2 = json.loads(text2)
            _normalize_object_attributes(obj2)
            validate(obj2, SCHEMA)
            return obj2
        except Exception as e:
            error_dict = {"obj": text2, "error": getattr(e, "message", None) or str(e)}
            with open(not_json_dir, "a", encoding="utf-8") as out_f:
                out_f.write(str(error_dict) + "\n")
            raise
