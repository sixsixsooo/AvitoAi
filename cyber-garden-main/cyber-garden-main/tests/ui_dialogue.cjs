// Exercise the actual browser script without adding frontend dependencies.
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
class Node {
  constructor(tag = 'div') { this.tag = tag; this.children = []; this.handlers = {}; this.classList = {add() {}}; this.value = ''; }
  append(...nodes) { for (const node of nodes) { node.parent = this; this.children.push(node); } }
  replaceChildren(...nodes) { this.children = []; this.append(...nodes); }
  get firstElementChild() { return this.children[0]; }
  cloneNode() { return new Node(this.tag); }
  addEventListener(type, callback) { this.handlers[type] = callback; }
  setAttribute() {}
  focus() {}
  remove() { if (this.parent) this.parent.children = this.parent.children.filter(n => n !== this); }
  replaceWith(node) { const p = this.parent; const i = p.children.indexOf(this); node.parent = p; p.children[i] = node; }
}
function all(node) { return [node, ...node.children.flatMap(all)]; }
function create() {
  const ids = new Map();
  function byId(id) { if (!ids.has(id)) ids.set(id, new Node()); return ids.get(id); }
  byId('evidence').append(new Node());
  byId('messages').append(byId('starter'));
  const requests = [];
  const replies = [];
  const context = vm.createContext({document: {getElementById: byId, createElement: tag => new Node(tag)}, Intl,
    fetch: async (path, options) => {
      if (path === '/api/catalog') return {ok: true, json: async () => ({total: 1, cities: ['Таганрог']})};
      requests.push(JSON.parse(options.body));
      const body = replies.shift();
      return {ok: true, json: async () => await body};
    }});
  vm.runInContext(fs.readFileSync('buyer/ui/app.js', 'utf8'), context);
  const state = {requirements: [{label: 'Город', value: 'Таганрог'}], eligible: [], budget_rub: 5000,
    preferences: [], trace: {model: 'offline', prompt_tokens: 1, completion_tokens: 1, elapsed_ms: 1},
    items: {t: {category: 'Стол', category_id: 'table', title: 'Стол', price_rub: 1000, city: 'Таганрог'}}};
  const ok = {...state, response: {status: 'ok', options: [{listing_ids: ['t'], total_rub: 1000, explanation: 'Проверено'}]}};
  return {byId, context, requests, replies, state, ok};
}
async function main() {
  const failures = [];
  async function test(name, run) { try { await run(); console.log('PASS', name); } catch (e) { failures.push(name); console.log('FAIL', name, e.message); } }
  await test('stale cards disappear while a new request is pending', async () => {
    const t = create(); t.replies.push(t.ok); await t.context.submit('Стол');
    let release; t.replies.push(new Promise(resolve => {release = resolve;}));
    const pending = t.context.submit('рецепт гуляша');
    const wasCleared = t.byId('results').children.length === 0;
    release({...t.state, items: {}, response: {status: 'clarify', questions: [{field: 'categories', text: 'Уточните товар'}]}});
    await pending;
    assert.equal(wasCleared, true);
    assert.equal(all(t.byId('results')).some(n => n.className === 'pane option'), false);
    assert.equal(t.requests[1].messages.at(-1).text, 'рецепт гуляша');
  });
  await test('no_match displays an explanation outside raw JSON', async () => {
    const t = create(); t.replies.push(t.ok); await t.context.submit('Стол');
    t.replies.push({...t.state, items: {}, response: {status: 'no_match', message: 'Подходящих ламп нет'}});
    await t.context.submit('Добавь лампу');
    assert.equal(all(t.byId('results')).some(n => n.tag === 'p' && n.textContent === 'Подходящих ламп нет'), true);
    assert.equal(all(t.byId('results')).some(n => n.className === 'pane option'), false);
  });
  await test('equal verified results are described honestly', async () => {
    const t = create(); t.replies.push(t.ok, t.ok);
    await t.context.submit('Стол'); await t.context.submit('Столик');
    assert.equal(all(t.byId('messages')).some(n => n.tag === 'p' && /не изменились/.test(n.textContent)), true);
  });
  await test('reset button clears history, cards and evidence', async () => {
    const t = create(); t.replies.push(t.ok); await t.context.submit('Стол');
    t.byId('reset').handlers.click();
    assert.equal(vm.runInContext('history.length', t.context), 0);
    assert.equal(t.byId('results').children.length, 0);
    t.replies.push(t.ok); await t.context.submit('Лампа');
    assert.equal(t.requests.at(-1).messages.length, 1);
  });
  await test('typed new dialogue starts an isolated request', async () => {
    const t = create(); t.replies.push(t.ok); await t.context.submit('Стол');
    t.replies.push({...t.state, items: {}, response: {status: 'clarify', questions: [{field: 'categories', text: 'Что приобрести?'}]}});
    await t.context.submit('новый диалог');
    assert.equal(t.requests.at(-1).messages.length, 1);
    assert.equal(t.byId('messages').children.filter(n => n.className === 'bubble user').length, 1);
  });
  function dataWith(t, options) {
    const items = {};
    for (const option of options) for (const id of option.listing_ids) {
      const category = id[0] === 'l' ? 'lamp' : id[0] === 'c' ? 'chair' : 'table';
      items[id] = {...t.state.items.t, category_id: category, category, title: id};
    }
    return {...t.state, items, response: {status: 'ok', options}};
  }
  function option(ids, total = 1000) { return {listing_ids: ids, total_rub: total, explanation: 'Проверено'}; }
  function textPresent(t, tag, text) { return all(t.byId('results')).some(n => n.tag === tag && n.textContent === text); }
  await test('lamp alternatives are individual listings', async () => {
    const t = create(); t.replies.push(dataWith(t, [option(['l1']), option(['l2']), option(['l3'])]));
    await t.context.submit('Лампа');
    assert.equal(textPresent(t, 'h2', 'Подходящие товары'), true);
    assert.equal(textPresent(t, 'p', 'Найдено 3 объявления'), true);
    assert.equal(all(t.byId('messages')).some(n => n.tag === 'p' && n.textContent?.startsWith('Нашёл подходящие объявления.')), true);
  });
  await test('one table is one listing', async () => {
    const t = create(); t.replies.push(dataWith(t, [option(['t1'])]));
    await t.context.submit('Только стол');
    assert.equal(textPresent(t, 'h2', 'Подходящие товары'), true);
    assert.equal(textPresent(t, 'p', 'Найдено 1 объявление'), true);
  });
  await test('table chair lamp remain a bundle', async () => {
    const t = create(); t.replies.push(dataWith(t, [option(['t1', 'c1', 'l1'])]));
    await t.context.submit('Стол, кресло и лампа');
    assert.equal(textPresent(t, 'h2', 'Подходящие комплекты'), true);
    assert.equal(textPresent(t, 'p', 'Найден 1 комплект'), true);
    assert.equal(all(t.byId('messages')).some(n => n.tag === 'p' && n.textContent?.startsWith('Подобрал комплект')), true);
  });
  await test('changed bundle receives a fresh result message', async () => {
    const t = create(); t.replies.push(dataWith(t, [option(['t1', 'c1'])]), dataWith(t, [option(['t2', 'c1'], 2000)]));
    await t.context.submit('Стол и кресло'); await t.context.submit('Другой стол');
    const texts = all(t.byId('messages')).filter(n => n.tag === 'p').map(n => n.textContent);
    assert.equal(texts.at(-1).startsWith('Подобрал комплект'), true);
    assert.equal(texts.at(-1).includes('не изменились'), false);
  });
  await test('unchanged results name goods and bundles separately', async () => {
    for (const [ids, noun] of [[['t1'], 'товары'], [['t1', 'c1', 'l1'], 'комплекты']]) {
      const t = create(); const data = dataWith(t, [option(ids)]); t.replies.push(data, data);
      await t.context.submit('Подбери'); await t.context.submit('Те же условия');
      const texts = all(t.byId('messages')).filter(n => n.tag === 'p').map(n => n.textContent);
      assert.equal(texts.at(-1).startsWith(`Подходящие ${noun} не изменились`), true);
    }
  });
  assert.equal(failures.length, 0, failures.join(', '));
}
main().catch(error => {console.error(error.message); process.exitCode = 1;});
