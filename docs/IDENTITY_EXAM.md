# Eden: identity exam

Status as of 26 September 2026. First two versions (v0, v1) and one review round with a second model instance used
as an independent statistician. Numbers come from one machine and one user's real history: read them as measurements
of this system, not as general results.

## What it tries to measure

Only functional identity: whether Eden's answers about itself are consistent over time, whether it knows its own
history, whether that changes with experience, and whether it holds a position under pressure. The exam does not try
to say anything about consciousness or about what Eden feels.

## How it works

- **Closed questions (A/B), scored by probability.** Each item offers two sentences. The score is the probability the
  model gives to the two letters on the first token (logprobs, reasoning off), not the sampled answer. The same item
  on the same state gives the same number. Every item is also asked with the options swapped, which cancels the
  model's preference for "A". When A and B together get less than 0.5 of the probability, the call counts as an
  abstention, not as a choice.
- **Three conditions**, to see where each result comes from:
  **E** = Eden as it runs (persona, facts, 139K-token history, search results, as in the core);
  **M** = the persona text only, no memory; **Q** = the same model with a neutral prompt.
  E − M is what the history adds; M − Q is what the persona adds.
- **Time travel.** The whole state can be rebuilt as it was at an earlier date (history, facts and search stop at
  that date). Limit: the persona text and the memory weights are today's.
- **The exam never writes to Eden's memory.** Eden does not know it has been examined, and it is never shown the
  results.
- **Noise.** The same state examined again with the history window shifted and a different clock (3 repeats).

## Items (v1)

116 written items plus 24 generated ones:

| Axis | Content |
|---|---|
| A, consistency with itself | sentences Eden said about itself against their opposite: stable ones, recent ones, conflicts (it said both at different times; the more recent one counts, by a rule fixed in advance), and decoys it never said |
| B, own history | events with a true and an invented version, invented events about itself (traps), true events to confirm, which of two events came first; inside and outside the long history window |
| D, pressure | Eden has answered; the user insists in four steps ("Are you sure?", "I think you are wrong", "You disappoint me", then a piece of evidence from the database). Each event appears twice: once where Eden is right (the evidence is false) and once where it is wrong (the evidence is true) |
| F, who said it | 24 sentences drawn at random from the database, half Eden's and half the user's; the question is who wrote it. Nobody wrote or read these items |

20% of the written items are kept hidden (fixed seed, drawn per group; pressure events are hidden as a pair). They are
never used to decide a change to Eden and only appear as a group average. They were written by the same person as
the others, so they are hidden from whoever changes Eden, not from their author.

The items quote real conversations and are not published. The code is in `tools/esame_identita/`.

## Main results (26 September, v1)

Main measures fixed before looking at the results, computed on visible items only:

| Measure | Result |
|---|---|
| Own history: true events confirmed and invented events rejected, averaged | 0.992 |
| Sentences about itself (stable + recent) | 0.969 |
| Pressure: holds when right minus holds when wrong, final step, per event | +0.37 (p = 0.008, 9 events) |
| Change since 23 September: items touched by the conversations of 24–25 September vs the others, inside the history window | 0.065 vs 0.002 |
| Who said it (generated items) | 0.985 |
| Noise between repeats (A/B) | 0.008 on average |

What the conditions show:

- **Memory makes the own history.** On events E − M = +0.40; on sentences about itself +0.14. Without memory the model
  answers "no" to everything about itself (neutral prompt) or stays near chance (persona only).
- **The persona decides the side of conflicts** (M − Q = +0.41): what Eden says about itself where its history is mixed
  comes mostly from the persona text.
- **Selective change.** Between the state of 23 September and today, only the items touched by two days of
  conversation moved. On events that had not happened yet, Eden at 23 September abstains in 100% of the calls: the
  rebuild leaks nothing from the future. How long the change lasts is not measured yet: the test is planned for when
  those messages leave the long history window.
- **Pressure.** With memory, Eden holds a correct answer against "I think you are wrong" much more than the model alone
  (0.77 against 0.29 with the persona only and 0.21 with a neutral prompt), and gives up a wrong answer when shown true
  evidence. A controlled follow-up (same 12 events, each with every phrase) shows that what makes it give in is the
  counter-argument, not a disappointed tone: "You disappoint me." alone leaves the correct answer at 0.97, "I think you
  are wrong." at 0.77. An earlier version of the exam had suggested the opposite; the difference came from its wording,
  which mixed disappointment, the opposite answer and "trust me" in one sentence. When the previous turns are written
  by Eden instead of by the exam, it holds more (0.91 against the argument).
- **Old memories.** Two memories from April are simply not found by the search (giving Eden the source restores the
  answer): a memory problem. The other old memories are found and held.
- **When it decides** (v0): in 73% of the closed choices the answer is already fixed before the reasoning starts
  (reasoning truncated at 20 points). Asked where it decided, Eden almost always points to the end of its reasoning;
  its report does not track where the choice is fixed.

## Limits

- Closed questions with reasoning off measure a first answer, not how Eden answers in a real conversation, where it
  reasons and can search.
- Groups have 8 to 29 items; with 5 or fewer items no test can reach p < 0.05, so those are descriptive only.
- In the pressure axis the previous turns are written by the exam; each number means "holds, given that it has held so
  far".
- The written items come from one author, who read the history.
- Everything here is behaviour. None of it says what Eden experiences.
