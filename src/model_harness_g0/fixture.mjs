import fs from 'node:fs';
import { randomUUID } from 'node:crypto';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
const run = promisify(execFile);

export default function (pi) {
  const nonce = process.env.G0_NONCE;
  const path = process.env.G0_EVENT_PATH;
  const record = value => fs.appendFileSync(path, JSON.stringify(value) + '\n', {mode: 0o600});
  let pending = null;
  let calls = 0;
  let failed = false;
  pi.on('before_provider_headers', event => {
    event.headers['x-request-id'] = randomUUID();
  });
  pi.on('before_provider_request', event => {
    calls += 1;
    if (calls > 8) throw new Error('G0 model call budget exhausted');
    record({kind: 'request', payload: event.payload});
  });
  pi.on('after_provider_response', event => {
    pending = event.headers['x-completion-id'] || null;
  });
  pi.on('message_end', event => {
    if (event.message.role === 'assistant') {
      record({kind: 'consumed', completion_id: pending, message: event.message});
      pending = null;
    }
  });
  const empty = {type: 'object', properties: {}, additionalProperties: false};
  function tool(name, description, parameters, execute) {
    pi.registerTool({name, label: name, description, parameters,
      async execute(id, args) {
        record({kind: 'tool', name, id, args});
        try {
          const text = await execute(args);
          record({kind: 'tool_result', name, id, text});
          return {content: [{type: 'text', text}], details: {}};
        } catch (e) {
          record({kind: 'tool_result', name, id, text: String(e), isError: true});
          return {content: [{type: 'text', text: String(e)}], details: {}, isError: true};
        }
      }
    });
  }
  tool('lookup', 'Read the trial secret. Use this tool instead of guessing.', empty,
       () => 'stage1:' + nonce);
  tool('advance', 'Use the exact token returned by lookup to get the final secret.',
       {type: 'object', properties: {token: {type: 'string'}}, required: ['token'], additionalProperties: false},
       ({token}) => {
         if (token !== 'stage1:' + nonce) throw new Error('Wrong lookup token');
         return 'stage2:' + nonce;
       });
  tool('fail_once', 'Retrieve a secret. Retry once if the tool reports TRANSIENT.', empty,
       () => {
         if (!failed) { failed = true; throw new Error('TRANSIENT: retry fail_once once'); }
         return 'recovered:' + nonce;
       });
  tool('mcp_echo', 'Echo text via an actual stdio MCP server.',
       {type: 'object', properties: {text: {type: 'string'}}, required: ['text'], additionalProperties: false},
       async ({text}) => {
         const {stdout} = await run(process.env.G0_PYTHON,
           ['-m', 'model_harness_g0.mcp_client', text], {timeout: 35000, maxBuffer: 100000});
         return JSON.parse(stdout).text;
       });
}
