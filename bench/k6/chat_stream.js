// Streaming chat completions at a fixed number of concurrent streams (closed model).
//
// The same script measures both paths, so client behaviour is identical:
//   gateway: TARGET_URL=http://gateway:8080/v1/chat/completions MODEL=fake API_KEY=...
//   direct:  TARGET_URL=http://fake-provider:8000/v1/chat/completions MODEL=fake-chat
//
// Metrics (milliseconds; k6 has no sub-millisecond clock, so values are whole ms):
//   ttft_ms    request start → first SSE event carrying content
//   stream_ms  request start → [DONE]
// Iterations that start during WARMUP are run but not recorded.
import sse from 'k6/x/sse';
import exec from 'k6/execution';
import { Counter, Rate, Trend } from 'k6/metrics';

const TARGET_URL = __ENV.TARGET_URL;
const MODEL = __ENV.MODEL;
const API_KEY = __ENV.API_KEY || '';
const VUS = parseInt(__ENV.VUS || '50', 10);
const WARMUP_S = parseInt(__ENV.WARMUP_S || '30', 10);
const DURATION_S = parseInt(__ENV.DURATION_S || '180', 10);
const PROMPT_TOKENS = parseInt(__ENV.PROMPT_TOKENS || '500', 10);
const SUMMARY_PATH = __ENV.SUMMARY_PATH || '/results/k6-summary.json';

const ttft = new Trend('ttft_ms');
const streamTime = new Trend('stream_ms');
const failed = new Rate('failed');
const completed = new Counter('completed');

export const options = {
  scenarios: {
    streams: {
      executor: 'constant-vus',
      vus: VUS,
      duration: `${WARMUP_S + DURATION_S}s`,
      gracefulStop: '30s',
    },
  },
  summaryTrendStats: ['avg', 'min', 'med', 'p(90)', 'p(95)', 'p(99)', 'max', 'count'],
};
// Note: xk6-sse opens a fresh HTTP transport per call, so every stream uses a new TCP
// connection. The direct baseline pays the same cost, so it cancels out of the comparison.

// About 4 bytes per token of English-like filler; the unique suffix defeats caching.
const FILLER = 'The quick brown fox jumps over the lazy dog while the gateway routes it. '.repeat(
  Math.ceil((PROMPT_TOKENS * 4) / 72),
).slice(0, PROMPT_TOKENS * 4);

export function setup() {
  return { startedAt: Date.now() };
}

export default function (data) {
  const recording = Date.now() >= data.startedAt + WARMUP_S * 1000;
  const body = JSON.stringify({
    model: MODEL,
    stream: true,
    stream_options: { include_usage: true },
    max_tokens: 256,
    messages: [
      { role: 'user', content: `${FILLER} [vu ${exec.vu.idInTest} iter ${exec.vu.iterationInScenario}]` },
    ],
  });
  const headers = { 'content-type': 'application/json' };
  if (API_KEY) headers.authorization = `Bearer ${API_KEY}`;

  const start = Date.now();
  let first = null;
  let done = false;
  let errored = false;
  const response = sse.open(TARGET_URL, { method: 'POST', body, headers, timeout: '120s' }, (client) => {
    client.on('event', (event) => {
      if (event.data === '[DONE]') {
        done = true;
        client.close();
        return;
      }
      // Cheap string checks instead of JSON.parse per chunk: k6's JS runtime would
      // otherwise become the bottleneck at 500 streams.
      if (first === null && event.data.indexOf('"content":"') !== -1 && event.data.indexOf('"content":""') === -1) {
        first = Date.now();
      } else if (event.data.indexOf('"error"') !== -1) {
        errored = true;
      }
    });
    client.on('error', () => {
      errored = true;
    });
  });

  const ok = response && response.status === 200 && done && !errored && first !== null;
  if (!recording) return;
  failed.add(!ok);
  if (ok) {
    completed.add(1);
    ttft.add(first - start);
    streamTime.add(Date.now() - start);
  }
}

export function handleSummary(data) {
  return { [SUMMARY_PATH]: JSON.stringify(data, null, 2) };
}
