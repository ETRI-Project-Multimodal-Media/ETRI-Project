import json
import re

from jsonschema import ValidationError, validate

SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["video_id", "event_id", "tags", "objects", "actors", "event", "LOD"],
    "properties": {
        "video_id": {"type": "string"},
        "event_id": {"type": "string", "pattern": r"^E_.+$"},
        "tags": {"type": "array", "items": {"type": "string"}},
        "objects": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["object_id", "name", "attributes"],
                "properties": {
                    "object_id": {"type": "string", "pattern": r"^O\d{3,}$"},
                    "name": {"type": "string"},
                    "attributes": {"type": "object", "additionalProperties": {"type": "string"}},
                },
            },
        },
        "actors": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["actor_id", "name", "ref_object", "role", "entity"],
                "properties": {
                    "actor_id": {"type": "string", "pattern": r"^A\d{3,}$"},
                    "name": {"type": "string"},
                    "ref_object": {"type": "string"},
                    "role": {"type": "string"},
                    "entity": {"type": "string"},
                },
            },
        },
        "event": {
            "type": "object",
            "additionalProperties": False,
            "required": ["event_id", "name", "time", "actors", "objects"],
            "properties": {
                "event_id": {"type": "string"},
                "name": {"type": "string"},
                "time": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["start", "end"],
                    "properties": {"start": {"type": "string"}, "end": {"type": "string"}},
                },
                "actors": {"type": "array", "items": {"type": "string"}},
                "objects": {"type": "array", "items": {"type": "string"}},
            },
        },
        "LOD": {
            "type": "object",
            "additionalProperties": False,
            "required": ["abstract_topic", "scene_topic", "summary", "implications"],
            "properties": {
                "abstract_topic": {"type": "array", "items": {"type": "string"}},
                "scene_topic": {"type": "string"},
                "summary": {"type": "string"},
                "implications": {"type": "string"},
            },
        },
    },
}


def _has_colon_outside_quotes(text: str) -> bool:
    in_string = False
    escape = False
    for ch in text:
        if escape:
            escape = False
            continue
        if ch == "\\" and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if not in_string and ch == ":":
            return True
    return False


def _fix_attribute_content(content: str) -> str:
    lines = content.splitlines(keepends=True)
    fixed_lines = []
    attr_counter = 1

    for line in lines:
        if not line.strip():
            fixed_lines.append(line)
            continue

        body = line.rstrip("\r\n")
        newline = line[len(body):]
        stripped = body.strip()
        if not stripped:
            fixed_lines.append(line)
            continue

        has_trailing_comma = stripped.endswith(",")
        base = stripped[:-1].strip() if has_trailing_comma else stripped

        if base and not _has_colon_outside_quotes(base):
            indent = body[: len(body) - len(body.lstrip())]
            try:
                value_obj = json.loads(base)
            except Exception:
                value_obj = base.strip('"')

            if isinstance(value_obj, (dict, list)):
                value_text = json.dumps(value_obj, ensure_ascii=False)
            else:
                value_text = str(value_obj).strip()

            limit = 512
            if len(value_text) > limit:
                value_text = value_text[: limit - 3] + "..."

            value_json = json.dumps(value_text, ensure_ascii=False)
            new_body = f'{indent}"attr_{attr_counter:03d}": {value_json}'
            if has_trailing_comma:
                new_body += ","
            fixed_lines.append(new_body + newline)
            attr_counter += 1
        else:
            fixed_lines.append(line)

    return "".join(fixed_lines)


def _find_matching_brace(text: str, open_brace_index: int) -> int:
    depth = 0
    in_string = False
    escape = False

    for idx in range(open_brace_index, len(text)):
        ch = text[idx]
        if escape:
            escape = False
            continue
        if ch == "\\" and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return idx
    return -1


def _fix_attributes_in_json(text: str) -> str:
    pattern = re.compile(r'"attributes"\s*:\s*\{', re.IGNORECASE)
    idx = 0
    result = text

    while True:
        match = pattern.search(result, idx)
        if not match:
            break

        content_start = match.end()
        closing_index = _find_matching_brace(result, match.end() - 1)
        if closing_index == -1:
            idx = content_start
            continue

        content = result[content_start:closing_index]
        fixed_content = _fix_attribute_content(content)
        result = result[:content_start] + fixed_content + result[closing_index:]
        idx = content_start + len(fixed_content)

    return result


def _normalize_object_attributes(obj: dict) -> None:
    objects = obj.get("objects")
    if not isinstance(objects, list):
        obj["objects"] = []
        return

    for item in objects:
        if not isinstance(item, dict):
            continue

        attrs = item.get("attributes")
        if isinstance(attrs, dict):
            normalized = {}
            for key, value in attrs.items():
                key_str = str(key)
                if isinstance(value, str):
                    normalized[key_str] = value
                elif isinstance(value, (int, float, bool)):
                    normalized[key_str] = str(value)
                elif value is None:
                    continue
                else:
                    normalized[key_str] = json.dumps(value, ensure_ascii=False)
            item["attributes"] = normalized
        elif isinstance(attrs, list):
            normalized = {}
            for idx, value in enumerate(attrs, start=1):
                normalized[f"attr_{idx:03d}"] = str(value)
            item["attributes"] = normalized
        elif isinstance(attrs, str):
            item["attributes"] = {"description": attrs}
        else:
            item["attributes"] = {}
