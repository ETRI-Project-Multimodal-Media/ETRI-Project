def _format_timestamp(value):
    if value is None:
        return ""
    if isinstance(value, (int, float)):
        formatted = f"{value:.2f}"
        if "." in formatted:
            formatted = formatted.rstrip("0").rstrip(".")
        return formatted or "0"
    return str(value)


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _iter_caption_segments(node, min_level=1, prefer_leaves=True):
    if not isinstance(node, dict):
        return

    children = node.get("children") or []
    level = node.get("level", 0)
    start = _format_timestamp(node.get("start_time"))
    end = _format_timestamp(node.get("end_time"))
    caption = (node.get("caption") or "").strip()
    is_leaf = not children

    for child in children:
        yield from _iter_caption_segments(child, min_level=min_level, prefer_leaves=prefer_leaves)

    if caption and start and end and level >= min_level and (is_leaf or not prefer_leaves):
        yield {"start": start, "end": end, "text": caption, "node": node, "level": level}


def _collect_segments(video_tree, min_level=0, prefer_leaves=False):
    segments = list(_iter_caption_segments(video_tree, min_level=min_level, prefer_leaves=prefer_leaves))
    segments.sort(key=lambda seg: (_safe_float(seg["start"]), _safe_float(seg["end"])))
    return segments
