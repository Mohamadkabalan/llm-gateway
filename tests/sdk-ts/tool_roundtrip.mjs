// A tool-call round trip with the official OpenAI TypeScript SDK through Switchboard to
// the fake Anthropic provider (alias `fake-claude`): stream a tool call, send the tool
// result back, and get the final answer. Run against `make up`:
//
//   OPENAI_BASE_URL=http://127.0.0.1:7080/v1 OPENAI_API_KEY=<dev key> npm test
import assert from 'node:assert/strict';
import OpenAI from 'openai';

const client = new OpenAI({ maxRetries: 0 });
const model = process.env.SWITCHBOARD_MODEL ?? 'fake-claude';
const tools = [
  {
    type: 'function',
    function: {
      name: 'get_weather',
      parameters: { type: 'object', properties: { location: { type: 'string' } } },
    },
  },
];
const messages = [{ role: 'user', content: "What's the weather in Paris?" }];

// 1. Streamed tool call.
const stream = await client.chat.completions.create({ model, messages, tools, stream: true });
let id = '';
let name = '';
let args = '';
let finish = null;
for await (const chunk of stream) {
  const choice = chunk.choices[0];
  if (!choice) continue;
  for (const call of choice.delta.tool_calls ?? []) {
    id ||= call.id ?? '';
    name ||= call.function?.name ?? '';
    args += call.function?.arguments ?? '';
  }
  finish = choice.finish_reason ?? finish;
}
assert.equal(finish, 'tool_calls');
assert.equal(name, 'get_weather');
assert.ok(id, 'tool call id');
const parsed = JSON.parse(args);
assert.equal(parsed.location, 'Paris');

// 2. The tool result goes back; the model answers in text.
messages.push(
  { role: 'assistant', content: null, tool_calls: [{ id, type: 'function', function: { name, arguments: args } }] },
  { role: 'tool', tool_call_id: id, content: '18C and sunny' },
);
const answer = await client.chat.completions.create({ model, messages, tools });
assert.equal(answer.choices[0].finish_reason, 'stop');
assert.ok(answer.choices[0].message.content);
console.log(`ok: ${model} tool round trip (${name} ${args}) -> "${answer.choices[0].message.content.trim()}"`);
