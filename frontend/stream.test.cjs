// Dependency-free protocol tests: node --test frontend/stream.test.cjs
const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const source = fs.readFileSync(path.join(__dirname, 'app.js'), 'utf8');
const parserSource = source.slice(source.indexOf('async function readEvents('), source.indexOf('async function sendMessage('));
const urlSource = source.slice(source.indexOf('function safeApiUrl('), source.indexOf('async function request('));
const documentSource = source.slice(source.indexOf('function documentIdentity('), source.indexOf('async function refreshDocuments('));
const context = vm.createContext({ TextDecoder, URL, window: { location: { origin: 'https://classroom.example' } } });
vm.runInContext(`${urlSource}\n${parserSource}\n${documentSource}`, context);

function responseFor(text, chunkSize = 1, onCancel) {
  const bytes = new TextEncoder().encode(text);
  let position = 0;
  return new Response(new ReadableStream({
    pull(controller) {
      if (position === bytes.length) { controller.close(); return; }
      controller.enqueue(bytes.slice(position, position + chunkSize));
      position = Math.min(bytes.length, position + chunkSize);
    },
    cancel: onCancel,
  }));
}

test('decodes byte-split UTF-8 tokens, CRLF, comments and EOF event', async () => {
  const events = [];
  await context.readEvents(responseFor(': heartbeat\r\n\r\nevent: meta\r\ndata: {"session_id":"123"}\r\n\r\nevent: token\r\ndata: {"text":"课程 🧠"}\r\n\r\nevent: done\r\ndata: {"text":"课程 🧠"}'), (type, data) => events.push([type, data]));
  assert.equal(events.length, 3);
  assert.equal(events[0][0], 'meta');
  assert.equal(events[0][1].session_id, '123');
  assert.equal(events[1][1].text, '课程 🧠');
  assert.equal(events[2][0], 'done');
});

test('joins multiline JSON data and resets event type for subsequent messages', async () => {
  const events = [];
  await context.readEvents(responseFor('event: sources\ndata: {"sources":\ndata: [{"id":"S1"}]}\n\ndata: {"text":"default event"}\n\n', 13), (type, data) => events.push([type, data]));
  assert.equal(events[0][0], 'sources');
  assert.equal(events[0][1].sources[0].id, 'S1');
  assert.equal(events[1][0], 'message');
});

test('rejects malformed JSON instead of presenting a successful answer', async () => {
  await assert.rejects(context.readEvents(responseFor('event: token\ndata: {broken}\n\n'), () => {}), /unreadable stream event/);
});

test('cancels the reader when an error event aborts consumption', async () => {
  let cancelled = false;
  const body = new ReadableStream({
    start(controller) { controller.enqueue(new TextEncoder().encode('event: error\ndata: {"message":"Unavailable"}\n\n')); },
    cancel() { cancelled = true; },
  });
  await assert.rejects(context.readEvents(new Response(body), (type, data) => { if (type === 'error') throw new Error(data.message); }), /Unavailable/);
  assert.equal(cancelled, true);
  assert.equal(body.locked, false);
});

test('authenticated fetch URL allowlist permits only same-origin API and exact audio routes', () => {
  assert.equal(context.safeApiUrl('/api/sessions?course_id=default'), 'https://classroom.example/api/sessions?course_id=default');
  assert.equal(context.safeApiUrl('/audio/' + 'a'.repeat(32) + '.wav'), 'https://classroom.example/audio/' + 'a'.repeat(32) + '.wav');
  for (const unsafe of ['https://other.example/api/steal', '//other.example/api/steal', '/audio/../admin', '/audio/not-a-real-id.wav', 'javascript:alert(1)', '/metrics', '/api/../../external']) {
    assert.throws(() => context.safeApiUrl(unsafe), /invalid API URL/);
  }
});

test('uses server document_id for document deletion and legacy id only as fallback', () => {
  assert.equal(context.documentIdentity({ document_id: 'server-id', id: 'different-id' }), 'server-id');
  assert.equal(context.documentIdentity({ id: 'legacy-id' }), 'legacy-id');
  assert.equal(context.documentIdentity({ title: 'No identifier' }), '');
});
