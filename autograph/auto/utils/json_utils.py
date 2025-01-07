import json

def is_json_serializable(obj):
    try:
        json.dumps(obj)
        return True
    except (TypeError, OverflowError):
        return False

def serialize_iterables_in_dict(root: dict):
    copy = {}
    for k, v in root.items():
        if type(v) == dict:
            copy[k] = serialize_iterables_in_dict(v)
        else:
            if type(v) in [tuple, set, map, filter]:
                copy[k] = list(v)
            else:
                copy[k] = v
    return copy