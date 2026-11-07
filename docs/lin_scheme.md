# The Lin-scheme baseline codebook

The dialogue-act scheme of Lin et al. (2022, *Future Generation Computer Systems* 127,
doi:10.1016/j.future.2021.09.001), Table 2, applied to our transcripts by **our LLM annotator**.

**This is a baseline using Lin's scheme. It is not a replication.** Lin's BERT classifier and
human-coded dialogues are not public, their outcome is a different one (a three-level expert
label of the dialogue), and their data is typed chat with images. Call it the "Lin-scheme
baseline" everywhere.

The annotation prompt is parsed from this file by `src/ttapub/taxonomy_lin.py`, so changing a
definition means editing §3 here. `tests/test_lin_scheme.py` holds Table 2 as literal text and
checks this file against it.

**Frozen after the pilot.** Nothing here may change after any test-split data has been looked at.

## 1. Sources, and what each one actually provides

Lin's Table 2 gives examples, not definitions, and cites Vail & Boyer (2014) for the scheme.
Reading that paper shows that it gives tag names, examples, frequencies and κ, **also without
prose definitions**. The only table of prose definitions in this lineage is one level up, for
the first-level tags, in the sister paper on the same corpus. So most second-level definitions
below have to be ours, and each row says so.

| key | source | gives |
|---|---|---|
| `L22-T2` | Lin et al. 2022, Table 2 (p. 198), read from the rendered page | tag, parent, role, example |
| `L22-P` | Lin et al. 2022, §3.2 prose (p. 197) | one-clause glosses of 7 tags |
| `VB14` | Vail & Boyer 2014, ITS, LNCS 8474, Tables 1–2 and §6, doi:10.1007/978-3-319-07221-0_24, [PDF from the authors' lab](https://intellimedia.ncsu.edu/wp-content/uploads/sites/42/Vail-ITS-2014.pdf) | flat 31-tag list with examples; one gloss (Directive) |
| `M13` | Mitchell, Ha, Boyer & Lester 2013, *IJLT*, Table 3, [PDF from the authors' lab](https://www.intellimedia.ncsu.edu/wp-content/uploads/sites/42/mitchell-ijlt-2013.pdf) | **prose definitions of the 13 prior tags, Lin's first level plus Other** |
| `L23` | Lin et al. 2023, AIED, [appendix B](https://github.com/jionghaolin/INFO) | more examples of the same second-level tags; no definitions |
| `OURS` | written here | marked in every row |

Checked and of no use for definitions: Lin's own appendix (`github.com/bertDA/BertDA`,
`DA_Appendix.pdf`), which holds sample dialogues only.

## 2. Structure, and what was checked against the page

* **12 first-level tags and 31 second-level tags, and the 31 already include Lin's 2 image
  tags** (Hint by Image, Request Feedback by Image). Counted from the rendered table:
  3+1+1+1+2+2+1+1+1+8+4+6 = 31. So **29 tags remain after the image tags go** (§5).
* **The tag-to-role mapping passes an arithmetic check against Lin's own feature counts.** §3.5
  of Lin reports 18 first-level features (6 of 12 shared between roles) and 41 second-level
  features (10 of 31 shared). The mapping in §3 gives exactly those: shared second-level
  Observation, Acknowledge, Correction, Extra Domain Question, Yes-No Answer, WH Answer, Extra
  Domain Answer, Explanation, Greeting, Extra Domain Other (10); shared first-level Hint,
  Acknowledge, Correction, Question, Answer, Statement (6). 31+10 = 41 and 12+6 = 18.
* The text layer garbles the merged Role cells and loses the last two rows' roles entirely. The
  roles below were read from the page image.
* **Lin's table differs from Vail & Boyer's in ways Lin does not fully state.** Vail & Boyer also
  have 31 tags, but a different 31. Lin merges Negative Feedback and its elaborated form into one
  tag, merges Other Feedback and its elaborated form into Lukewarm Feedback, and adds 2 image
  tags: 31 − 2 + 2 = 31. The equal counts are a coincidence. Lin's first-level labels are the
  prior tags of `M13` without Other. Vail & Boyer's `Questions (QQ)` is Lin's Operational Question
  (Lin says so in the table note). We follow Lin throughout.

## 3. The scheme

One row per second-level tag. `role`: T, S or T&S. A T&S tag becomes two separate columns, one
per role. The first-level tag is fixed by the second-level tag, so it is derived and not asked of
the annotator (§4.3).

| code | first | Lin's name | role | Lin's example (Table 2) | definition | source of the definition |
|---|---|---|---|---|---|---|
| `INFORMATION` | HINT | Information | T | It can be any one of the cards in the deck. | Tutor gives advice to help the student proceed with the task, by stating a piece of information the student can use, without telling them the next step to take. | First clause `M13` (Hint); rest OURS |
| `OBSERVATION` | HINT | Observation | T&S | We have 80. | The speaker points out something that is already there in the problem or the work so far, without explaining why it holds. | OURS, from `L22-T2`, `VB14` and `L23` examples |
| `DIRECTIVE` | DIRECTIVE | Directive | T | Check this definition. | Tutor explicitly tells the student the next step to take; a bottom-out hint. | `M13` verbatim |
| `ACKNOWLEDGE` | ACKNOWLEDGE | Acknowledge | T&S | Alright! | Acknowledgement of a previous utterance; conversational grounding. The speaker expresses agreement with or acknowledgement of the other person. | First sentence `M13` verbatim; second `L22-P` reworded |
| `EVALUATION_QUESTION` | REQUEST_CONFIRMATION | Evaluation Question | T | Does that make sense? | Tutor requests confirmation or grounding from the student: asks whether what was just said or done is understood. | First clause `M13` (Request Confirmation); rest OURS |
| `CONFIRMATION_QUESTION` | REQUEST_FEEDBACK | Confirmation Question | S | Would the answer be 30? | Student requests an assessment of their performance or work from the tutor, by proposing an answer or step and asking whether it is right. | First clause `M13` (Request for Feedback); rest OURS |
| `GENERAL_POSITIVE_FEEDBACK` | POSITIVE_FEEDBACK | General Positive Feedback | T | Correct! | Tutor gives a positive assessment of the student's performance, with no statement of what specifically was good. | First clause `M13` (Positive Feedback); rest OURS, from `VB14` FP vs FPE |
| `ELABORATED_POSITIVE_FEEDBACK` | POSITIVE_FEEDBACK | Elaborated Positive Feedback | T | Your formula for period is correct! | Tutor gives a positive assessment of the student's performance and says what specifically was good. | First clause `M13` (Positive Feedback); rest OURS, from `VB14` FP vs FPE |
| `NEGATIVE_FEEDBACK` | NEGATIVE_FEEDBACK | Negative Feedback | T | No, it is incorrect. | Tutor gives a negative assessment of the student's performance. | `M13` verbatim |
| `LUKEWARM_FEEDBACK` | LUKEWARM_FEEDBACK | Lukewarm Feedback | T | Almost correct, but the sign is missing. | Tutor gives an assessment that has both positive and negative elements. | `M13` verbatim |
| `CORRECTION` | CORRECTION | Correction | T&S | *We will | The speaker repairs a slip in their own previous utterance. In these spoken transcripts that is a misspeaking repaired, standing in for the typos of typed chat. Correcting the other person is not this tag. | First sentence OURS from `M13`, `L22-P`; spoken adaptation OURS (§4.5) |
| `DIRECTION_QUESTION` | QUESTION | Direction Question | S | How do I do that? | Student asks what to do next or how to proceed with the task. | OURS, from `L22-T2`, `VB14` examples |
| `INFORMATION_QUESTION` | QUESTION | Information Question | S | What are the units for W? | Student asks for a fact or an explanation about the content, not for what to do next and not for a check of their own answer. | OURS, from `L22-T2`, `VB14`, `L23` examples |
| `PROBING_QUESTION` | QUESTION | Probing Question | T | How many options can it be? | Tutor asks a question that prompts the student's critical thinking about the problem at hand. | `L22-P` reworded |
| `OPEN_QUESTION` | QUESTION | Open Question | T | What do you think we could try next? | Tutor asks a question with no single expected answer, encouraging the student to reason and reflect. | Second clause `L22-P` reworded; first OURS |
| `FACTUAL_QUESTION` | QUESTION | Factual Question | T | What is the value of x? | Tutor asks a question requiring the student to recall or state a specific fact, value or concept. | `L22-P` reworded |
| `OPERATIONAL_QUESTION` | QUESTION | Operational Question | T | Any questions on this? | Tutor asks about how the session should go or whether the student has questions or needs, rather than about the content. | OURS, from `L22-T2`, `VB14` (QQ), `L23` examples |
| `READY_QUESTION` | QUESTION | Ready Question | T | Are you ready to begin? | Tutor asks whether the student is ready to begin or move on. | OURS, from examples |
| `EXTRA_DOMAIN_QUESTION` | QUESTION | Extra Domain Question | T&S | How are you doing today? | A question about something outside the mathematics and the task, such as how the person is, their school or their day. | OURS, from examples |
| `YES_NO_ANSWER` | ANSWER | Yes-No Answer | T&S | Yes, that would be very helpful. | An answer to a yes-no question from the other person. | Parent `M13` (Answer); rest OURS |
| `WH_ANSWER` | ANSWER | WH Answer | T&S | It is 6. | An answer to a what, which, when, where, who, why or how question from the other person. | Parent `M13` (Answer); rest OURS |
| `READY_ANSWER` | ANSWER | Ready Answer | S | Yes, I'm ready. | Student answers a question about whether they are ready. | Parent `M13` (Answer); rest OURS |
| `EXTRA_DOMAIN_ANSWER` | ANSWER | Extra Domain Answer | T&S | I'm good. | An answer to a question about something outside the mathematics and the task. | Parent `M13` (Answer); rest OURS |
| `EXPLANATION` | STATEMENT | Explanation | T&S | The straight line is the line on the bottom. | A statement about the content or the work that explains or describes it, and is not an answer to a question just asked. | OURS, parent `M13` (Statement) |
| `GREETING` | STATEMENT | Greeting | T&S | Hello! | Opening or closing the conversation: hello, welcome, goodbye, have a good day. | `L22-P` keyword rule; closings from `VB14` example; OURS |
| `EXTRA_DOMAIN_OTHER` | STATEMENT | Extra Domain Other | T&S | Welcome to use this app! | A statement about something outside the mathematics and the task, including the platform, the audio or screen, and small talk. | OURS, from `L22-T2`, `VB14`, `L23` examples |
| `REASSURANCE` | STATEMENT | Reassurance | T | No problem, I will help you. | Tutor comforts or reassures the student about the difficulty, the mistake or the time available. | OURS, from `L22-T2`, `VB14`, `L23` examples |
| `UNDERSTANDING` | STATEMENT | Understanding | S | Ok, got it. | Student says that they now understand. | OURS, from `L22-T2`, `VB14`, `L23` examples |
| `NOT_UNDERSTANDING` | STATEMENT | Not Understanding | S | I don't know why. | Student says that they do not understand or do not know. | OURS, from `L22-T2`, `VB14`, `L23` examples |
| `UNCODABLE` | NONE | (ours, not in Lin) | T&S | [unclear] | Nothing intelligible: only an [unclear] marker, or a fragment too broken to tell what the speaker is doing. A sentence with an [unclear] marker and readable words is tagged on those words. Never a way to avoid a hard decision. | OURS, a departure (§4.4) |

### Boundary rules (OURS; Lin publishes none)

| pair | rule |
|---|---|
| `ACKNOWLEDGE` vs `YES_NO_ANSWER` | "Yes" or "okay" is `YES_NO_ANSWER` only if the sentence just before it from the other person was a yes-no question. Otherwise it is `ACKNOWLEDGE`. |
| `ACKNOWLEDGE` vs `GENERAL_POSITIVE_FEEDBACK` | "Okay" or "right" that only signals the tutor heard is `ACKNOWLEDGE`. It is feedback only if it evaluates what the student just said. |
| `INFORMATION` vs `EXPLANATION` | `INFORMATION` gives the student something to use on the next step. `EXPLANATION` accounts for or describes something already on the table. |
| `DIRECTIVE` vs `INFORMATION` | `DIRECTIVE` tells the student to do something. `INFORMATION` only states something. |
| `PROBING_QUESTION` vs `FACTUAL_QUESTION` | `FACTUAL_QUESTION` has an answer the student recalls or reads off. `PROBING_QUESTION` needs the student to work something out. |
| `PROBING_QUESTION` vs `OPEN_QUESTION` | `PROBING_QUESTION` has a right answer. `OPEN_QUESTION` invites an approach or a reason. |
| `EVALUATION_QUESTION` vs `OPERATIONAL_QUESTION` | "Does that make sense?" checks understanding of what was said and is `EVALUATION_QUESTION`. "Any questions?" asks for the student's needs and is `OPERATIONAL_QUESTION`. |
| `GREETING` vs `EXTRA_DOMAIN_OTHER` | Lin's rule assigns sentences with "hello" or "welcome" to Greeting, yet Lin's own example of Extra Domain Other is a "welcome". Here a greeting of the person is `GREETING`; a statement about the platform is `EXTRA_DOMAIN_OTHER`. |

## 4. Decisions

### 4.1 Unit of annotation

Lin coded sentences, and Vail & Boyer "sentential and phrasal" segments; this repo's pipeline codes
turns. TSL transcripts are spoken but carry punctuation (80% of student and 88% of tutor turns,
measured on 300 training sessions, end in `.`, `?` or `!`), so a sentence split is possible.
Whether it is reliable is tested on the pilot against §6, fixed here before the pilot was run. If
it fails, the fallback is turn level, reported as a departure from Lin.

### 4.2 Two levels and Lin's 5% rule

Lin gave second-level tags only to sentences whose first-level tag exceeded 5% of sentences. **We
do not apply that rule at annotation time: every sentence receives a second-level tag.** The rule
was about human coders' cost, it depends on a corpus prevalence not known in advance, and our own
prevalence gate does the filtering (`≥ 300` positives). Lin's 41 second-level features also
cover tags whose parents are below 5%. The pilot reports which first-level tags would have
passed.

### 4.3 First level is derived

The annotator outputs the second-level tag only. Each second-level tag has exactly one parent, so
the first-level tag follows, and asking for both could only produce inconsistent pairs.

### 4.4 `UNCODABLE`

Lin's table has no residual tag and guided decoding forces a choice, so a bare `[unclear]` would be
pushed into a real tag and inflate it. This repo has already measured that failure once (the
residual codes of `taxonomy_v3.md`). `UNCODABLE` is the narrowest fix: **it gets no feature column
and is excluded from the rate's denominator**, because "not measured" and "the tag did not occur"
are different statements. Its prevalence is reported. `M13`'s Other tag, which Lin dropped, is not
restored.

### 4.5 Spoken data

`CORRECTION` was defined on typos, and Vail & Boyer excluded it from their analysis as an artefact of
instant messaging. It is kept and adapted to spoken repairs, and it may well fail the gate.

### 4.6 Main-arm columns and kappa

The main arm has one rate per role-tag that clears the gate, plus one per first-level tag with at
least two second-level children for that role. A first-level tag with one child would be the same
column twice. If every tag passes that is 47 columns.

Kappa stays 25 to match the clean57 arms, but the unit is now usually the sentence and sessions hold
more sentences than turns. The shrinkage is therefore relatively weaker than on turns. This was
fixed before any features were built and is not tuned.

## 5. Image tags: dropped

Hint by Image and Request Feedback by Image describe a sent image, in an app where students
photograph problems. **Our train transcripts are ASR of speech** and contain no image events: a
scan of 300 training sessions found 5 of 79,550 utterances matching an image or link pattern (the
full-split count is in §7). They cannot occur, so they are dropped, which leaves **39 role-tag
columns: 23 tutor + 16 student**. Eedi is typed chat. It is test data, so it did not inform this
decision. The feature run reports the Eedi count of image markers after this file was frozen, and
any such line would be `UNCODABLE`.

## 6. Criteria for sentence segmentation, set before the pilot

Sentence level is used if all of these hold on the pilot's training sessions; otherwise turn level.

1. Of 200 randomly drawn turns read by hand by the authors, at least 90% have every sentence boundary acceptable
   (no split inside a clause, no two clearly separate acts left joined).
2. At most 5% of sentences are 30 words or longer (run-ons).
3. At most 5% of sentences are `UNCODABLE` (segments too broken to code).
4. Parse failures at or under the stage's 5% threshold, and no systematic off-vocabulary output.

## 7. Record

Filled in as measured. See the run manifest (reproduce/runs.tsv) for the runs.

| check | value |
|---|---|
| Image or link pattern, whole training split | 79 of 6,139,854 utterances (`results/lin-features`). Dropping the image tags (§5) costs nothing measurable. |
| Pilot round 1 (`results/lin-pilot`, 50 training sessions) | **Failed criterion 3.** `UNCODABLE` was 4.9% of tutor and **12.5% of student** sentences against a 5% limit. 68% of the student cases were fragments holding only `[unclear]`, cut loose by the splitter; others had readable words ("[unclear] Yes."). Other criteria: run-ons 0.6% / 1.0%; no parse failures or off-vocabulary output; about 1% of sentences omitted by the model. Hand reading of 200 turns found about 3% with a wrong boundary (a missing full stop joining two acts), plus the bare fragments. |
| Changes after round 1, criteria unchanged | Bare `[unclear]` fragments join the neighbouring sentence; `UNCODABLE`'s definition and prompt rule 5 say readable words are tagged on those words; prompt rule 4 names `[Speaker:...]` markers; each chunk asks for exactly one object per number. No outcome or test data had been seen. |
| Pilot round 2 (`results/lin-pilot2`, 50 fresh training sessions, offset 50) | **Sentence level kept.** (1) Hand reading of 200 turns by the authors: about 96% with every boundary acceptable; the failures are a missing full stop joining two acts. (2) Run-ons 0.6% tutor, 0.7% student. (3) `UNCODABLE` **3.3% of all sentences as the criterion is worded (2.1% tutor, 5.9% student).** The student share alone is above 5%, by less than the 2.8 points that are turns holding only `[unclear]`, which turn level could not code either. Some sentences with readable words ("[unclear] Yeah.") are still `UNCODABLE`, a limit of the annotator. Round 1 was judged by role, where it failed; it also fails overall, at 7.4%. (4) No parse failures, no off-vocabulary output, 3 of 26,178 sentences omitted. |
| Eedi image markers, after freeze | 0 of 27,131 Eedi and 2 of 1,948,847 TSL test utterances (`results/lin-features`). Counted only after the codebook froze. |
| Full pass coverage (`results/lin-features`) | Every session with text labelled, both splits. Sentences labelled: tutor 99.99%, student 99.97%. `UNCODABLE`: train tutor 2.1%, student 6.0%; test tutor 2.4%, student 5.1%. |
| Prevalence gate (`results/lin-features/gates.csv`) | All 47 columns clear 300 positives; none fails. Median split-half reliability (within-session, block 10): tutor 0.60, student 0.46. |
