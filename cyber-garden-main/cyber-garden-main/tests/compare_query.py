"""Compare local parsing, raw Gemma and reconciliation against the same annotated turns."""

import statistics
import sys
import time

from buyer.conversation import ConversationManager
from buyer.model import ModelError, extract
from tests.test_conversation import CASES


def main():
    manager = ConversationManager()
    local_count = gemma_count = combined_count = gateway_errors = fast = product_count = 0
    durations = []
    prompt_tokens = []
    completion_tokens = []
    print("| Сценарий | Локальный | Gemma | Гибрид | Продуктовый маршрут |", flush=True)
    print("| --- | --- | --- | --- | --- |", flush=True)
    for index, (name, messages, gold) in enumerate(CASES):
        started = time.monotonic()
        local = manager.local(messages)
        local_ok = local.requirements == gold and not local.unresolved
        local_count += local_ok
        fast += not local.needs_model
        try:
            model, trace = extract(messages, full_history=True, use_anchors=False)
        except ModelError as error:
            gateway_errors += 1
            hybrid_ok = False
            gemma_status = hybrid_status = "ошибка шлюза или формата"
            print(f"Вызов Gemma, сценарий «{name}»: {error}", file=sys.stderr, flush=True)
        else:
            gemma_ok = model.requirements == gold
            hybrid_ok = manager.merge(local, model).requirements == gold
            gemma_count += gemma_ok
            combined_count += hybrid_ok
            gemma_status, hybrid_status = ("да" if ok else "нет" for ok in (gemma_ok, hybrid_ok))
            durations.append(trace.elapsed_ms)
            if trace.prompt_tokens is not None and trace.completion_tokens is not None:
                prompt_tokens.append(trace.prompt_tokens)
                completion_tokens.append(trace.completion_tokens)
        route = "Gemma" if local.needs_model else "локальный"
        product_count += hybrid_ok if local.needs_model else local_ok
        print(f"| {name} | {'да' if local_ok else 'нет'} | {gemma_status} | {hybrid_status} | {route} |",
              flush=True)
        if index + 1 < len(CASES):
            time.sleep(max(0, 4.2 - (time.monotonic() - started)))
    print(f"Точных совпадений из {len(CASES)}: локальный {local_count}, "
          f"Gemma {gemma_count}, гибрид {combined_count}.", flush=True)
    print(f"Локальный маршрут: {fast}/{len(CASES)}. Ошибок вызова: {gateway_errors}.", flush=True)
    print(f"Гибрид с вызовом только по необходимости: {product_count}/{len(CASES)}.", flush=True)
    if durations:
        print(f"Медиана успешных вызовов Gemma: {statistics.median(durations):g} мс.", flush=True)
    if prompt_tokens:
        print(f"Медиана токенов: вход {statistics.median(prompt_tokens):g}, "
              f"выход {statistics.median(completion_tokens):g}.", flush=True)
    return bool(gateway_errors)


if __name__ == "__main__":
    raise SystemExit(main())
