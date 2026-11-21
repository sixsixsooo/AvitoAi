"""Entrypoint for the organizers' stdin/stdout contract."""

import argparse
import json
import signal
import sys
from pathlib import Path

from buyer.contracts import decode_json, read_catalog, read_messages
from buyer.conversation import ConversationManager
from buyer.model import ModelError, extract
from buyer.selection import select


def _timeout(_signal, _frame):
    raise ModelError("Превышено время выполнения запроса.")


def main():
    parser = argparse.ArgumentParser(description="Подбор комплекта по контракту ЮФУ 1.1.")
    parser.add_argument("--catalog", required=True, type=Path, help="Путь к JSON-каталогу")
    args = parser.parse_args()

    try:
        request = read_messages(decode_json(sys.stdin.buffer.read()))
        listings = read_catalog(decode_json(args.catalog.read_bytes()))
        query, _ = ConversationManager().resolve(request, extract, require_model=True)
        attribute_questions = query.attribute_questions() if not query.questions() else ()
        if query.decision_questions():
            response = {"status": "clarify", "questions": list(query.decision_questions())}
        elif attribute_questions:
            # The official schema has no attribute-specific question fields.
            # Keep the supported categories and ask about their requirements
            # using the permitted composition field, rather than ignoring a bound.
            response = {"status": "clarify", "questions": [{"field": "categories", "text":
                        "Уточните требования к товарам комплекта. " + " ".join(attribute_questions)}]}
        else:
            response = select(listings, query.requirements, query.preferences).response
    except ModelError as error:
        print(f"run: {error}", file=sys.stderr)
        return 2
    except (ValueError, OSError, RecursionError):
        print("run: не удалось прочитать запрос или каталог по контракту 1.1.", file=sys.stderr)
        return 2

    sys.stdout.buffer.write((json.dumps(response, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8"))
    return 0


if __name__ == "__main__":
    if hasattr(signal, "SIGALRM"):
        signal.signal(signal.SIGALRM, _timeout)
        signal.alarm(115)
    raise SystemExit(main())
