# Eden 2: decisions and measurements

Status as of 23 September 2026. This document is in English; the other technical docs are in Italian.

Eden 2 is a rebuild of Eden, not a new project. The goal is that Eden remembers what was said to it and does not make things up. The first measurements showed that a very long context is not enough for that on its own, and that the memory needs literal storage, search and dated facts. Everything below was measured on one machine, with one user's real history, so read the numbers as indications and not as benchmarks.

## Plan

| # | Step | Status |
|---|---|---|
| 1 | Measurements: speed, memory recall, search, voice | done |
| 2 | Migrate everything Eden has lived into one SQLite file | done |
| 3 | Memory exam: about 80 questions taken from the real history, answers checked against the source messages. It decides the model and the memory setup | in progress |
| 4 | New core: prompt, loop, tools | not started |
| 5 | Search as a tool the model calls, with citations that are checked | not started |
| 6 | Sleep: facts with their source, open threads; a view of what Eden knows about the user | not started |
| 7 | Voice, UI, access from a phone | not started |
| 8 | Proactive messages, and removal of what is no longer needed | not started |

## Setup used for the measurements

- RTX 3090 Ti 24 GB runs the language model. RTX 5060 Ti 8 GB is for voice and listening. 16 GB of RAM.
- `llama-server` from llama.cpp (build 11029) in router mode, Qwen3.8-27B in Q4_K_M, 4-bit KV cache, flash attention, a draft of 2 tokens (MTP speculative decoding), one slot.
- Test history: the last ~140,000 tokens of my real conversation archive (1,613 exchanges, 2 May to 3 September 2026).

## Measurements

### Speed of the language model (reasoning off)

| Context | Prompt reading | Generation |
|---|---|---|
| 2,500 tokens | 840 tok/s | 39 tok/s |
| 21,000 tokens | 1,320 tok/s | 52 tok/s |
| 62,000 tokens | 1,120 tok/s | 47 tok/s |
| 140,000 tokens | 840 tok/s | 33–45 tok/s |

- The first read of 140,000 tokens takes 172 s (275 s if the model also has to be loaded).
- With the prefix cache, a new question on the same history takes 1–4 s (139,800 tokens reused). A chat that grows turn by turn takes 1–3 s per turn.
- Reasoning has to be switched off explicitly (`chat_template_kwargs.enable_thinking`). Otherwise the model reasons silently: on a small test model, 1,335 tokens and 12 s instead of 3 tokens and 0.08 s.

### Recall from a long history

9 real questions from the history: 8 facts and 1 trap (something that was never said). Two builds of the same model: a community variant and the official one.

| Method | Community variant | Official |
|---|---|---|
| Long history only | 6/9 | 7/9 |
| Keyword search only (FTS5, 12 exchanges) | 7/9 | 7/9 |
| Long history + search results at the end of the message | 9/9 (3–4 s) | 7/9 |
| Long history + reasoning on | 9/9 (4–21 s) | not run |

- The sample is small: a difference of 1–2 points is noise. The choice of the model is left to the memory exam (step 3).
- The community variant mixes English into 8 of 49 answers; the official build into 0 of 27.
- With a prompt that offered the way out "if it is not there, say it is not there", the community variant fell to 3/9: it used the way out instead of searching.

### Search over the whole archive

2,475 exchanges, 11 questions, 5 of them worded differently from the source text. A hit means the right exchange is in the top 12.

| Method | Hits |
|---|---|
| Keywords (FTS5, BM25) | 4/11 |
| Meaning (Qwen3-Embedding-0.6B) | 7/11 |
| Hybrid (reciprocal rank fusion) | 5/11 |
| Hybrid + reranker (Qwen3-Reranker-0.6B) | 6/11, +3.4 s |

- Strong paraphrases fail with every method. So search is given to the model as a tool, and the model rewrites the query.
- Embeddings on the CPU: 22 ms per query, 0.18 s to index a new exchange. They stay on the CPU and leave the 5060 Ti to voice and listening.
- The reranker gave no gain and was dropped.

### Voice and listening

| Engine | First audio | Speed | VRAM |
|---|---|---|---|
| `qwen-tts` (official), 0.6B / 1.7B | none (no streaming) | 0.3–0.4× real time | 2.7 / 4.6 GB |
| `faster-qwen3-tts` (CUDA graph), 1.7B | 0.41 s | 2.1× | 4.7 GB |
| `faster-qwen3-tts`, 0.6B | 0.37 s | 2.4× | 2.8 GB |

- I chose the 1.7B after listening to both. The 0.6B once did not stop (150 s of audio for one sentence), so a length limit is needed in production.
- Intelligibility: Whisper recognises 94–100% of the generated words.
- A bug found in my local Eden 1 setup: the reference text given to the voice model did not match the reference audio. With a mismatched text and audio, Qwen3-TTS produced nonsense words. The text has to be the exact transcript of the clip (in this repository `mechanisms/fish_tts.py` holds a placeholder for it).
- Listening with faster-whisper large-v3-turbo (int8): 0.43–0.53 s for 10–12 s of audio, 1.1 GB.
- Search + listening + voice all on the 5060 Ti need 7.67 of 8.15 GB and listening slows down to 12 s. The chosen split: voice and listening on the 5060 Ti (about 6.9 GB with the desktop), search on the CPU.
- Environment: `faster-qwen3-tts` 0.4.0 needs `qwen-tts-hf`, `transformers==5.15.1` (5.17 breaks it) and a CUDA build of torch.

### Cache saved to disk

Not usable yet. With hybrid models, saving the slot cache loses the restore points ([llama.cpp issue 25913](https://github.com/ggml-org/llama.cpp/issues/25913); a fix is in [PR 26004](https://github.com/ggml-org/llama.cpp/pull/26004), open at the time of writing). To try again when it is merged.

## Decisions

### Memory

- A long history (about 140K tokens) sits in the prefix of the prompt, plus a search tool. Search combines keywords (FTS5), meaning (embeddings) and a date filter. The model calls it and rewrites the query itself.
- Search results go at the end of the current message, so the cached prefix stays valid (measured: 3–4 s per answer).
- Questions about dates ("when did we talk after 29 May") cannot be answered by word search, so they need a calendar tool: days with a conversation, and the pauses between them.
- Reasoning is off in the chat and on only for questions about the past and for the sleep step. Not at the highest levels.
- The first read of the history takes minutes, so it has to happen before the user writes.
- If the same model server is shared with another tool that asks for a different model, the cache is lost and reading the history again takes 3–4 minutes. Two slots on the same model are under consideration.

### Storage: one SQLite file

The database is not published. The tables:

| Table | Content |
|---|---|
| `messaggi` (messages) | literal text, timestamp, role (`user`, `assistant`, `sistema`, `pensiero`), session, source, id of the original exchange, original fields as JSON |
| `messaggi_fts` | FTS5 index, accent-insensitive, kept up to date by triggers |
| `fatti` (facts) | subject, key, value, valid from and valid until, the message it came from, confidence, a flag for facts not to be mentioned unless the user asks |
| `giudizi` (judgments) | thumbs up, thumbs down, corrections and scores per message, kept for future fine-tuning |
| `embedding` | message id, model, float32 vector in the sqlite-vec layout. Empty for now |

- The migration only reads the original files. It compares the SHA-256 of 64 files before and after, builds the database from scratch, and refuses to overwrite an existing one unless told to.
- The old facts (extracted with text rules, some of them wrong) were not copied. The sleep step will rebuild them from the messages, each with its source.
- The migration found 26 exchanges that were in the graph database and in the session files but not in the JSONL archive, and recovered them. The archive numbering also has 63 gaps that could not be recovered.
- The migration script and the exam questions contain personal facts and are not published.

### Sleep

- The heavy memory work happens when the user is away, with reasoning on (the idea is called sleep-time compute, from Letta).
- Each fact keeps the message it came from and a validity interval. The idea comes from Zep and Graphiti, kept here in SQLite.
- Open threads come from the same step.

### Voice and listening

- Qwen3-TTS 1.7B through `faster-qwen3-tts`, streaming sentence by sentence; listening with faster-whisper large-v3-turbo.
- Studied, not measured yet: Smart Turn v3 (end-of-turn detection, BSD-2, 12 ms on the CPU) to allow interruptions; emotion2vec+ to read the tone of the user's voice; a tone instruction in the cloned voice, to link Eden's internal state to how it sounds.

## Discarded

- End-to-end full-duplex speech models: they do not use Eden's own model and memory, and they do not handle Italian. A chain with end-of-turn detection already gives natural interruptions.
- Persona vectors and activation steering: tried in May (two attempts), they did not work.
- MEMENTO: needs the model to be retrained, and helps long reasoning rather than conversation memory.
- Fish Audio S2 Pro: 9 GB of weights, it does not fit on the 5060 Ti.
- Reranker: no gain measured.

## Open questions

- Which build of the model to use: the choice is left to the memory exam.
- Whether one model server can serve Eden and another tool at the same time without losing the cache.
- How the search tool will behave when the model rewrites the query badly: this belongs to step 5.
