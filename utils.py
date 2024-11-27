import json
import hashlib

def serialize_dict(dict_to_serialize):
    args_json_str = json.dumps(dict_to_serialize,sort_keys=True)
    args_hex_digest = hashlib.sha256(args_json_str.encode()).hexdigest()
    return args_hex_digest