// Browser upload recovery logic, executed with a small DOM/fetch test double. No dependencies.
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

async function panel(resumeState, lookupStatus = 200) {
  const elements = new Map();
  function element() {
    return {
      value: '', files: [], textContent: '', classList: {toggle() {}},
      append() {}, replaceChildren() {}, addEventListener() {},
      showModal() {}, close() {},
    };
  }
  const document = {
    getElementById(id) {
      if (!elements.has(id)) elements.set(id, element());
      return elements.get(id);
    },
    createElement: element,
  };
  const id = 'a'.repeat(32), newId = 'b'.repeat(32);
  const file = {name: 'data.jsonl', size: 5, lastModified: 1, slice: () => new Blob(['hello'])};
  const storage = new Map([['ft-upload-dataset', JSON.stringify({id, signature: JSON.stringify([
    {path: file.name, size: file.size, modified: file.lastModified},
  ])})]]);
  const calls = [];
  let reads = 0, uploaded = false;
  const asset = {id, name: 'Dataset', kind: 'dataset', bytes: 5, state: resumeState};
  const context = vm.createContext({
    document, Option: function(text, value) { this.text = text; this.value = value; },
    localStorage: {getItem: key => storage.get(key), setItem: (key, value) => storage.set(key, value), removeItem: key => storage.delete(key)},
    setInterval() {}, setTimeout: callback => queueMicrotask(callback),
    fetch: async (url, options = {}) => {
      const method = options.method || 'GET';
      calls.push({url, method});
      let data, status = 200;
      if (url === '/api/session') data = {token: 'token', slurm_available: true, architectures: ['olmo3_7b'], chunk_bytes: 8};
      else if (url === '/api/assets') data = {assets: [{...asset, id: uploaded ? newId : id, state: 'ready'}]};
      else if (url === '/api/runs') data = {runs: []};
      else if (url === '/api/uploads/' + id) {
        status = lookupStatus;
        data = status === 200 ? {asset: {...asset, state: reads++ ? 'ready' : resumeState}, files: [{offset: 5}]} : {error: 'lookup failed'};
      } else if (url === '/api/uploads' && method === 'POST') data = {asset: {...asset, id: newId, state: 'uploading'}};
      else if (url === '/api/uploads/' + newId) data = {asset: {...asset, id: newId}, files: [{offset: 0}]};
      else if (url.endsWith('/files/0') && method === 'PUT') data = {offset: 5};
      else if (url.endsWith('/complete') && method === 'POST') {uploaded = true; data = {asset: {...asset, id: newId, state: 'ready'}};}
      else throw new Error('Unexpected request: ' + method + ' ' + url);
      return {ok: status < 400, status, json: async () => data};
    },
  });
  vm.runInContext(fs.readFileSync(path.join(__dirname, '../dashboard/launch.js'), 'utf8'), context);
  await new Promise(setImmediate); // finish page initialization
  document.getElementById('dataset-files').files = [file];
  document.getElementById('dataset-name').value = 'Dataset';
  await vm.runInContext("upload('dataset')", context);
  return {calls, storage, elements};
}

test('lost completion response reuses a ready upload without resending files', async () => {
  const {calls, storage, elements} = await panel('ready');
  assert.equal(calls.filter(call => call.method !== 'GET').length, 0);
  assert.equal(storage.has('ft-upload-dataset'), false);
  assert.match(elements.get('dataset-status').textContent, /previous upload completed/);
});

test('reconnecting during hashing waits for the same upload', async () => {
  const {calls, elements} = await panel('checking');
  assert.equal(calls.filter(call => call.method !== 'GET').length, 0);
  assert.equal(calls.filter(call => call.url === '/api/uploads/' + 'a'.repeat(32)).length, 2);
  assert.match(elements.get('dataset-status').textContent, /previous upload completed/);
});

test('temporary lookup failure cannot silently create a duplicate upload', async () => {
  const {calls, storage, elements} = await panel('uploading', 500);
  assert.equal(calls.filter(call => call.method !== 'GET').length, 0);
  assert.equal(storage.has('ft-upload-dataset'), true);
  assert.equal(elements.get('dataset-status').textContent, 'lookup failed');
});

test('a deliberately removed upload can be uploaded again', async () => {
  const {calls, storage, elements} = await panel('uploading', 404);
  assert.equal(calls.filter(call => call.url === '/api/uploads' && call.method === 'POST').length, 1);
  assert.equal(calls.filter(call => call.method === 'PUT').length, 1);
  assert.equal(storage.has('ft-upload-dataset'), false);
  assert.match(elements.get('dataset-status').textContent, /Ready/);
});
