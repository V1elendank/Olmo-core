# Data formats

One dataset = one file (or a folder of same-type files): `.jsonl`, `.json`, `.csv`, `.parquet`.
Keep the same column names within a dataset; column names are detected from the first row.

| format | needs | loss on | example |
|---|---|---|---|
| `chat` | a list of messages: `{"role","content"}` (or ShareGPT `{"from","value"}`) | assistant turns | `chat.jsonl`, `chat_sharegpt.jsonl` |
| `prompt_response` | a prompt column + a response column (+ optional `input`) | the response | `prompt_response.jsonl`, `alpaca.jsonl`, `prompt_response.csv` |
| `text` | a text column | every token | `text.jsonl` |

Auto-detected column names (override with `*_column:` in the YAML):
messages: `messages, conversations, conversation, chat, dialog`;
prompt: `prompt, instruction, question, query, input_text`; extra input: `input, context`;
response: `response, output, answer, completion, chosen`; text: `text, content, document, body`.

Check what the model will actually see (trainable parts marked) in the control panel's Data tab,
or in `$OLMO_ROOT/data/<key>/manifest.json` after `python finetune/ft.py prepare-data CONFIG`.
