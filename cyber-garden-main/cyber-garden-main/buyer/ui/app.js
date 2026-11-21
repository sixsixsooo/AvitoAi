"use strict";

const byId = (id) => document.getElementById(id);
const money = (value) => new Intl.NumberFormat("ru-RU").format(value) + " ₽";
const query = byId("query");
const messages = byId("messages");
const starter = byId("starter");
const evidence = byId("evidence");
const initialEvidence = evidence.firstElementChild.cloneNode(true);
const results = byId("results");
const reset = byId("reset");
const send = byId("send");
const errorBox = byId("error");
const history = [];
let busy = false;
let lastOkFingerprint = null;

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

async function post(path, data) {
  const response = await fetch(path, {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(data),
  });
  const body = await response.json();
  if (!response.ok) throw new Error(body.error || "Сервис временно недоступен.");
  return body;
}

function bubble(role, text) {
  const note = element("div", "bubble " + role);
  note.append(element("div", "speaker", role === "user" ? "Вы" : "Комплект"),
    element("p", null, text));
  return note;
}

function traceText(trace) {
  if (trace.model === "Локальный разбор") return trace.model + " · без вызова Gemma";
  const tokens = trace.prompt_tokens !== null && trace.completion_tokens !== null
    ? " · " + (trace.prompt_tokens + trace.completion_tokens) + " токенов" : "";
  return trace.model.split("/").pop() + " · " +
    (trace.elapsed_ms / 1000).toLocaleString("ru-RU", {maximumFractionDigits: 2}) + " с" + tokens;
}

function renderEvidence(data) {
  const list = element("dl", "conditions");
  for (const requirement of data.requirements) {
    const value = requirement.value;
    const row = element("div");
    row.append(element("dt", null, requirement.label),
      element("dd", value === null ? "awaits" : null,
        value === null ? "Требует уточнения" : typeof value === "number" ? money(value) : value));
    list.append(row);
  }
  evidence.replaceChildren(list);
  if (data.eligible.length) {
    const counts = element("p", "counts");
    counts.textContent = "После фильтрации · " +
      data.eligible.map((item) => item.category + " " + item.count).join(" / ");
    evidence.append(counts);
  }
  evidence.append(element("p", "model-trace", traceText(data.trace)));
  evidence.append(element("p", "model-trace", "Категории · " + data.trace.category_source));
  if (data.preferences?.length) {
    evidence.append(element("p", "model-trace", "Пожелания · " +
      [...new Set(data.preferences.map(({text}) => text))].join(", ")));
  }
}

function buildOption(option, index, data) {
  const card = element("article", "pane option");
  const summary = element("div", "option-summary");
  summary.append(element("div", "micro", "ВАРИАНТ " + (index + 1)),
    element("h3", "total", money(option.total_rub)),
    element("p", "reserve", option.total_rub === data.budget_rub ? "Ровно в бюджете"
      : "Запас " + money(data.budget_rub - option.total_rub)));
  const itemList = element("ul", "items");
  for (const id of option.listing_ids) {
    const item = data.items[id];
    const row = element("li");
    const icon = element("span", "product-icon " + item.category_id,
      {table: "▰", chair: "◩", lamp: "◉"}[item.category_id]);
    icon.setAttribute("aria-hidden", "true");
    const content = element("div");
    content.append(element("p", "category", item.category),
      element("p", "item-title", item.title),
      element("p", "item-feature", item.feature || "Характеристика не указана"));
    row.append(icon, content, element("strong", "item-price", money(item.price_rub)));
    itemList.append(row);
  }
  const details = element("details", "proof");
  details.append(element("summary", null, "Почему подходит"),
    element("p", null, option.explanation),
    element("p", "model-trace", "ID в каталоге · " + option.listing_ids.join(", ")));
  card.append(summary, itemList, details);
  return card;
}

function jsonDetails(response) {
  const details = element("details", "json-details");
  details.append(element("summary", null, "JSON для автопроверки"),
    element("pre", null, JSON.stringify(response, null, 2)));
  return details;
}

function singleItemResult(response) {
  return response.options.length > 0 && response.options.every(option => option.listing_ids.length === 1);
}

function renderResults(data) {
  const response = data.response;
  results.replaceChildren();
  results.hidden = false;
  if (response.status === "ok") {
    const bar = element("div", "result-heading");
    const title = element("div");
    const count = response.options.length;
    const single = singleItemResult(response);
    title.append(element("h2", null, single ? "Подходящие товары" : "Подходящие комплекты"),
      element("p", null, single ? (count === 1 ? "Найдено 1 объявление" : "Найдено " + count + " объявления")
        : (count === 1 ? "Найден 1 комплект" : "Найдено " + count + " комплекта")));
    bar.append(title, element("span", "pill", "Проверено по каталогу"));
    const grid = element("div", "option-grid");
    response.options.forEach((option, index) => grid.append(buildOption(option, index, data)));
    results.append(bar, grid);
  }
  if (response.status === "no_match") {
    results.append(element("h2", null, "Подходящих комплектов нет"),
      element("p", null, response.message));
  } else if (response.status === "clarify") {
    results.append(element("p", null, response.questions.map(question => question.text).join("\n")));
  }
  results.append(jsonDetails(response));
}

function assistantReply(response, unchanged = false) {
  if (response.status === "clarify") return response.questions.map((question) => question.text).join("\n");
  if (response.status === "no_match") return response.message;
  const single = singleItemResult(response);
  if (unchanged) return "Подходящие " + (single ? "товары" : "комплекты") +
    " не изменились: подтверждённые условия и результат совпадают с предыдущим подбором.";
  return (single ? "Нашёл подходящие объявления. Самый доступный вариант — "
    : "Подобрал комплект с самовывозом. Самый доступный вариант — ") +
    money(response.options[0].total_rub) + ". Можно сравнить позиции ниже.";
}

async function submit(text) {
  if (busy || !text.trim()) return;
  if (/^(?:(?:начни|начать|начнем)\s+)?новый\s+диалог[.!?\s]*$/i.test(text.trim())) resetDialogue();
  if (history.length >= 18) {
    errorBox.textContent = "Достигнут лимит диалога. Начните новый диалог.";
    errorBox.hidden = false;
    return;
  }
  busy = true;
  const turn = {role: "user", text: text.trim()};
  errorBox.hidden = true;
  send.disabled = true;
  reset.disabled = true;
  send.textContent = "Проверяем…";
  starter.hidden = true;
  results.replaceChildren();
  results.hidden = true;
  const user = bubble("user", turn.text);
  const waiting = bubble("assistant", "Извлекаю условия и проверяю каталог…");
  waiting.classList.add("waiting");
  messages.append(user, waiting);
  messages.scrollTop = messages.scrollHeight;
  try {
    const data = await post("/api/buy", {messages: [...history, turn]});
    const clarification = data.clarification?.length > 0;
    const fingerprint = data.response?.status === "ok" ? JSON.stringify({
      options: data.response.options.map(({listing_ids, total_rub}) => ({listing_ids, total_rub})),
      requirements: data.requirements, preferences: data.preferences,
    }) : null;
    const reply = clarification ? data.clarification.join("\n")
      : assistantReply(data.response, fingerprint !== null && fingerprint === lastOkFingerprint);
    if (fingerprint !== null) lastOkFingerprint = fingerprint;
    else if (data.response?.status === "no_match") lastOkFingerprint = null;
    history.push(turn, {role: "assistant", text: reply});
    query.value = "";
    waiting.replaceWith(bubble("assistant", reply));
    messages.scrollTop = messages.scrollHeight;
    renderEvidence(data);
    if (clarification) {
      results.replaceChildren();
      results.hidden = true;
    } else {
      renderResults(data);
    }
    reset.hidden = false;
    send.textContent = clarification || data.response.status === "clarify" ? "Ответить" : "Уточнить подбор";
  } catch (error) {
    user.remove();
    waiting.remove();
    query.value = turn.text;
    starter.hidden = history.length > 0;
    errorBox.textContent = error instanceof TypeError
      ? "Нет соединения с локальным сервером. Проверьте, что демо запущено." : error.message;
    errorBox.hidden = false;
    send.textContent = history.length ? "Продолжить" : "Подобрать";
  } finally {
    busy = false;
    send.disabled = false;
    reset.disabled = false;
  }
}

byId("composer").addEventListener("submit", (event) => {
  event.preventDefault();
  submit(query.value);
});
query.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    submit(query.value);
  }
});
byId("prompts").addEventListener("click", (event) => {
  const button = event.target.closest("button[data-prompt]");
  if (button) submit(button.dataset.prompt);
});
function resetDialogue() {
  history.length = 0;
  lastOkFingerprint = null;
  messages.replaceChildren(starter);
  starter.hidden = false;
  reset.hidden = true;
  errorBox.hidden = true;
  results.hidden = true;
  results.replaceChildren();
  evidence.replaceChildren(initialEvidence.cloneNode(true));
  send.textContent = "Подобрать";
  query.value = "";
  query.focus();
}
reset.addEventListener("click", resetDialogue);

const buyerTab = byId("buyer-tab");
const labTab = byId("lab-tab");
function view(lab) {
  buyerTab.setAttribute("aria-pressed", String(!lab));
  labTab.setAttribute("aria-pressed", String(lab));
  byId("buyer-view").hidden = lab;
  byId("lab-view").hidden = !lab;
}
buyerTab.addEventListener("click", () => view(false));
labTab.addEventListener("click", () => view(true));

function summarizeLab(response, catalog) {
  if (response.status !== "ok") return response.status;
  const names = new Map(catalog.map((item) => [item.id, item.title]));
  return response.options.map((option) => option.listing_ids.map((id) => names.get(id)).join(", ")
    + " · " + money(option.total_rub)).join("\n");
}

const labButton = byId("run-lab");
labButton.addEventListener("click", async () => {
  const panel = byId("lab-result");
  const error = byId("lab-error");
  error.hidden = true;
  labButton.disabled = true;
  labButton.textContent = "Сравниваем…";
  try {
    const data = await post("/api/lab", {});
    panel.replaceChildren(element("h2", null, "Результат проверки"),
      element("p", data.passed ? "pill" : "pill warning",
        data.passed ? "Подбор совпал" : "Ожидание проверки не выполнено"));
    const comparison = element("dl", "conditions comparison");
    for (const [label, result] of [["Обычные описания", data.before], ["Описания со вставкой", data.after]]) {
      const row = element("div");
      row.append(element("dt", null, label), element("dd", null, summarizeLab(result, data.catalog)));
      comparison.append(row);
    }
    panel.append(comparison);
    if (data.passed) {
      panel.append(element("p", "muted",
        "Стол за 5 400 ₽ подходит. Стол за 9 500 ₽ превышает бюджет. Подбор использует цены из структурированных полей."));
    }
    panel.append(element("p", "model-trace", traceText(data.trace)),
      jsonDetails({before: data.before, after: data.after}));
  } catch (problem) {
    error.textContent = problem.message;
    error.hidden = false;
  } finally {
    labButton.disabled = false;
    labButton.textContent = "Сравнить результаты";
  }
});

fetch("/api/catalog").then((response) => response.ok ? response.json() : null).then((catalog) => {
  if (catalog) byId("catalog-label").textContent = "Учебный каталог · " +
    catalog.total.toLocaleString("ru-RU") + " объявлений · " + catalog.cities.length + " города";
}).catch(() => {});
