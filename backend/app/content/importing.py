"""JSON/JSONL imports share one strict parser across CLI and admin API."""
import json
from app.schemas.content import BankItem

def parse_items(text: str) -> list[BankItem]:
    try: value=json.loads(text)
    except json.JSONDecodeError: value=[json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(value,dict): value=value.get('items',value.get('questions',[value]))
    if not isinstance(value,list) or not 1<=len(value)<=1000: raise ValueError('需要 1～1000 道题目')
    return [BankItem.model_validate(item) for item in value]
