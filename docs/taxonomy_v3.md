# Annotation codebook

The codebook the annotation prompts are built from: 41 tutor moves and 16 student signals.
`src/ttapub/taxonomy_v3.py` parses §2-§4 when it builds a prompt, so a definition is edited here or
nowhere. Section numbers follow the full specification, so some are skipped.

## 2. Tutor pool — 41 codes

Source key: **NTO** = published NTO taxonomy, wording verbatim; **INT** = tutor interviews; **DATA** = close reading of transcripts by the authors, reported as
`CLOSE`; **RES** = residual, which is a consequence of single-label coding and not an origin
(see §4).

### 2.1 NTO — tutoring support (5)

| code | definition (verbatim) | src |
|---|---|---|
| `ASKING_TO_CLARIFY_CONTEXT` | Tutor asks the student about the problem to understand the context. | NTO |
| `PROBING_PRIOR_KNOWLEDGE` | Tutor probes about student's prior knowledge | NTO |
| `PROBING_UNDERSTAND` | Tutor questions whether student understood the statement made or step(s) done. | NTO |
| `CORRECTING_OWN_ERROR` | Tutor explicitly recognizes a mathematical error they made. | NTO |
| `STRATEGIZING` | Tutor discusses plans with the student to complete the problem. | NTO |

NTO's own distinction applies: *probing* = a categorical or yes/no question ("Are you with me?");
*prompting* = an open-ended question.

### 2.2 NTO — learning support (15), ordered high → low student engagement

| code | definition (verbatim) | src |
|---|---|---|
| `PROMPTING_RELATED_CONCEPTS` | Tutor prompts the student to make connections to related concepts without giving detailed explanations of the related concepts. | NTO |
| `PROMPTING_ALTERNATIVE_REPRESENTATION` | Tutor asks the student to represent the idea in a different form. | NTO |
| `PROMPTING_SELF_EXPLANATION` | Tutor prompts the student to explain actions or knowledge. | NTO |
| `PROMPTING_NEXT_STEP` | Tutor prompts the student to produce any step without adding new information. | NTO |
| `PROMPTING_SELF_CORRECTION` | Tutor prompts the student to correct their own mistakes without giving away the right way. | NTO |
| `FEEDBACK_CORRECT` | Tutor indicates that the student has provided a correct response. | NTO |
| `FEEDBACK_INCORRECT` | Tutor indicates that the student has provided an incorrect response. | NTO |
| `FEEDBACK_NEUTRAL` | Tutor comments without confirming accuracy or inaccuracy. | NTO |
| `REVOICING` | Tutor says what the student said in different ways. | NTO |
| `RESTATING` | Tutor says what the student said verbatim. | NTO |
| `GIVING_HINT` | Tutor provides a suggestion to guide the student toward a solution without revealing the answer. | NTO |
| `GIVING_EXAMPLE` | Tutor uses an example that is different from the student's problem. | NTO |
| `EXPLAINING_CONCEPTUAL` | Tutor gives instruction on concepts or facts. | NTO |
| `EXPLAINING_PROCEDURAL` | Tutor gives instruction about the algorithmic steps to solve the problem. | NTO |
| `GIVING_ANSWER` | Tutor provides the solution without including the solution steps. | NTO |

The NTO definitions are the **published** wording, verbatim; `tests/test_nto_verbatim.py`
checks that they are.

### 2.3 NTO — social-emotional & motivational (7) and logistical (1)

| code | definition (verbatim) | src |
|---|---|---|
| `BUILDING_RAPPORT` | Tutor engages in relationship-building conversation, including when the tutor humanizes themselves. | NTO |
| `ENCOURAGING` | Tutor makes encouraging statements to students. | NTO |
| `ASKING_FEELING` | Tutor asks for the student's feelings. | NTO |
| `VALIDATING_FEELING` | Tutor validates the student's feelings. | NTO |
| `PRAISING_PROCESS` | Tutor praises the student's process of solving the problem. | NTO |
| `PRAISING_OUTCOME` | Tutor praises the student's outcome of student work. | NTO |
| `PRAISING_TRAITS` | Tutor praises the student's characteristic. | NTO |
| `TECHNICAL_ISSUES` | Tutor makes a comment related to technical problems of the tutoring context. | NTO |

Interview 2's "praise effort on partial steps" is `PRAISING_PROCESS`, so it is now covered by NTO.

### 2.4 Ours — interview codes (7)

| code | definition | interview | distinct from |
|---|---|---|---|
| `RELEASE_HANDOFF` | Hands the current problem over for the student to attempt alone, after the tutor has been doing or leading it ("Now you try this one"). The defining feature is the transfer of who is working, not the question mark. | 2, 3 (we-do → you-do) | `PROMPTING_NEXT_STEP` asks for one step; handoff transfers the whole problem |
| `TRANSFER_PROBE` | Offers a DIFFERENT problem of about the SAME difficulty testing the same idea ("Same method, but with 3/8 this time"). Not a repeat, not a harder follow-on. | 2 (near-transfer check) | `GIVING_EXAMPLE` is the tutor illustrating; here the student must solve |
| `SIMPLIFYING_TO_SUBPROBLEM` | Replaces the current problem with a smaller or easier version ("Forget the fractions — what's 4 times 3?"). The tutor must change what is being worked on. | 1 | `GIVING_HINT` gives information about the existing problem |
| `MOTIVATING_RELEVANCE` | States where the idea is used or why it matters ("This is how shops work out the sale price"). About purpose, not illustration. | 1 | `GIVING_EXAMPLE` illustrates mechanics |
| `SUMMARIZING_PROGRESS` | Aggregates what has been established across several turns ("So we've got the numerator, and we said the denominator was 8…"). | 2 (recap when stuck) | `REVOICING`/`RESTATING` cover one turn |
| `OFFERING_CHOICE` | Hands the student control over WHAT to work on ("Which one do you want to try?"). Not a rhetorical "shall we?", not a choice of method. | 2 | `STRATEGIZING` plans how; this chooses what |
| `NORMALIZING_DIFFICULTY` | Explicitly licenses not-knowing or being stuck; makes the difficulty a property of the problem, not the student ("That's a really tricky one" / "Loads of people get that wrong"). | 1, 3 (making not-knowing safe) | see the boundary rule below |

**Boundary rule for the three affective codes** (the one place the codebook adds overlap risk):
`NORMALIZING_DIFFICULTY` is about the **problem** ("this is hard for everyone");
`VALIDATING_FEELING` responds to a feeling the **student expressed** ("I get that it's frustrating");
`ENCOURAGING` is a forward-looking push ("You can do this"). A turn doing two of these takes the one
that is its main purpose (§3).

### 2.5 Ours — data-derived codes (3)

| code | definition | distinct from |
|---|---|---|
| `GUIDING_SESSION` | Directs the flow of the lesson rather than the mathematics: announcing what comes next, moving between slides or tasks, allocating time. | `STRATEGIZING` is a plan for solving the problem |
| `EXPLAINING_TOOL` | Instructs on the platform rather than the mathematics: where to write, how to use a feature. | `TECHNICAL_ISSUES` is something broken |
| `WORKED_DEMONSTRATION` | Works the CURRENT problem through aloud to model the method. | `EXPLAINING_PROCEDURAL` describes steps in general; `GIVING_EXAMPLE` uses a different problem |

### 2.6 Residuals (3)

| code | definition |
|---|---|
| `TUTOR_UNINTELLIGIBLE` | The turn is [unclear] or [inaudible]. A transcription failure, not a move. |
| `TUTOR_OTHER` | Pedagogically meaningful, but no code fits. Reports a taxonomy gap. |
| `TUTOR_NO_MOVE` | Nothing pedagogically notable. Fallback only. |

## 3. Assigning one label per turn (both pools)

**No priority order.** A global ranking would decide every overlap in favour of whichever codes
we put on top, inflating their rates and building our own assumptions into the counts. Instead:

1. **Primary function.** Label what the turn is *mainly for*. *"That's a tricky one — let's just
   do 4 × 3 first"* is mainly a change of problem → `SIMPLIFYING_TO_SUBPROBLEM`; the reassurance is
   secondary.
2. **Pairwise boundary rules** for codes whose definitions border each other. The "distinct from"
   column in §2 and the affective rule in §2.4 are the starting set:

   | pair | rule |
   |---|---|
   | `SIMPLIFYING_TO_SUBPROBLEM` vs `GIVING_HINT` | did the tutor change *what* is being worked on? → simplifying; information about the same problem → hint |
   | `WORKED_DEMONSTRATION` vs `EXPLAINING_PROCEDURAL` vs `GIVING_EXAMPLE` | current problem solved aloud / steps in general / a different problem |
   | `RELEASE_HANDOFF` vs `PROMPTING_NEXT_STEP` | the whole problem handed over / one step asked for |
   | `TRANSFER_PROBE` vs `GIVING_EXAMPLE` | the student must solve it / the tutor illustrates |
   | `SUMMARIZING_PROGRESS` vs `REVOICING` / `RESTATING` | several turns aggregated / one turn reworded / one turn repeated |
   | `NORMALIZING_DIFFICULTY` vs `VALIDATING_FEELING` vs `ENCOURAGING` | about the problem / about a feeling the student expressed / a push forward |
   | `PROBING_UNDERSTAND` vs `PROMPTING_SELF_EXPLANATION` | yes/no check ("got it?") / open "how did you…?" |
   | `STRATEGIZING` vs `GUIDING_SESSION` | plan for solving the problem / flow of the lesson |
   | student `SELF_CORRECTS` vs answer codes | the student revises their own earlier answer → self-corrects, even if the new answer is bare |
   | student `EXPLAINS_WHY` vs `SHOWS_METHOD` | a reason or concept / the steps taken; both present → the one the turn mostly consists of |
   | student `HEDGED_ANSWER` vs `ASKS_VERIFICATION` | "Is it 12?" offering an answer → hedged answer; "is that right?" about an answer already given → asks verification |
   | `FEEDBACK_CORRECT` vs `PRAISING_OUTCOME` | the turn says the ANSWER is right ("yes, 24") → feedback; it says the STUDENT did well ("brilliant job") → praising. "Well done, that's 24" does both: the primary function is the evaluation, so feedback |
   | `PRAISING_PROCESS` / `PRAISING_OUTCOME` vs `GUIDING_SESSION` | praise followed by a move on ("great, now slide 4") is `GUIDING_SESSION` only if the praise is a formulaic discourse marker ("okay good, so…"); if it names what the student did, it is praising |
   | `FEEDBACK_CORRECT` vs `PROMPTING_NEXT_STEP` | a turn that confirms and then asks for the next step is `PROMPTING_NEXT_STEP`: the request for work is what shapes what the student does next |
   | student `INSIGHT` vs `STUDENT_NO_MOVE` | a discourse marker alone ("oh okay", "aha", "oh right") is `STUDENT_NO_MOVE`, however enthusiastic. `INSIGHT` requires the student to SAY what they now understand ("ohh, so the 6 is tens") |
   | student `EXPLAINS_WHY` vs `CONFIDENT_ANSWER` / `SHOWS_METHOD` | stating a fact about the problem ("the number has 6 tens") is an ANSWER, not a justification. `EXPLAINS_WHY` needs a reason or a causal link — because, so, since, a rule being invoked |
   | student `STUDENT_OTHER` and mis-attributed speech | some turns carry the other speaker's words (the corpus's speaker labels are imperfect). If a turn tagged student is plainly the tutor talking, use `STUDENT_OTHER` — do not try to force a student code onto it |

3. **Second code when torn (pilot only).** The annotator records an optional `also` code when a
   turn genuinely serves two functions. It is never used as a label; it measures which pairs need
   a rule. Any pair torn often in the pilot without a rule gets one before the full run.

## 4. Student pool — 16 codes (flat, agreed 2026-10-13)

Four student codes (`SELF_CORRECTS`, `EXPRESSES_NOT_KNOWING`, `ASKS_CONCEPTUAL`,
`ASKS_VERIFICATION`) come from close reading of transcripts, the same origin as the §2.5
codes. There is no external source to cite for any of them.

**Three origins, covering all 57.** Every code is published (`NTO`), came from asking tutors
(`INT`), or came from reading transcripts closely (`CLOSE`) — which covers codes added when
annotations are checked against the turns they describe, since that is also reading
transcripts.

**Residual is a property of a code, not an origin**. They were **introduced during human annotation**, on finding that without
them turns with no pedagogical content were absorbed by substantive codes and inflated
their rates. So their origin is close reading like any other
code found that way, and `residual` is tracked separately as what the code **is**: it absorbs
turns rather than asserting a construct, and no pedagogical claim may rest on one.

They are not inert. `STUDENT_OTHER` is 6.5% of student turns and overwhelmingly tutor speech
mis-tagged by diarisation, so its rate doubles as a mis-attribution detector; without the
residual it would have been scattered across the student signals instead, where it would have
been invisible.

| code | definition | src |
|---|---|---|
| `SELF_CORRECTS` | Revises their own earlier answer without being prompted ("wait, no — it's 24"). | DATA, defined via the §3 boundary rule |
| `INSIGHT` | Genuine moment of getting it ("ohh, so that's why"); not a polite "oh okay". | INT 1, 3 (relief as the tell) |
| `EXPLAINS_WHY` | Justifies with a reason or concept ("because the angles add to 180"). | INT |
| `SHOWS_METHOD` | Describes the steps taken ("I did the brackets first"). | INT |
| `HEDGED_ANSWER` | Bare answer marked as uncertain ("I think 12?"). | INT |
| `CONFIDENT_ANSWER` | Bare answer stated plainly or assertively ("12."). | INT |
| `EXPRESSES_NOT_KNOWING` | "I don't know", "I'm not sure". A wrong answer alone is NOT this (correctness is annotated separately). | DATA |
| `ASKS_CONCEPTUAL` | Asks why or how something works. | DATA |
| `ASKS_VERIFICATION` | Checks an answer or step ("is that right?"). | DATA |
| `FRUSTRATION` | Distress at the task ("ugh", "I can't do this", giving up); the student must express the feeling. | INT 1, 3 |
| `OFF_TASK` | Talk unrelated to the session. | new |
| `STUDENT_SOCIAL` | Greetings, small talk. | RES |
| `STUDENT_TECHNICAL` | Connection or audio. | RES |
| `STUDENT_UNINTELLIGIBLE` | [unclear] / [inaudible]. | RES |
| `STUDENT_OTHER` | Meaningful, but no code fits. | RES |
| `STUDENT_NO_MOVE` | Fallback ("okay", "mm"). | RES |

Interview student concepts deliberately **not** coded: negative
self-efficacy (0.9% of sessions, fails the prevalence gate), curiosity/extension (3.1%),
goal orientation (rarely stated), disengagement (moved to timestamp features).

Known consequence of a single label: a feeling expressed inside a turn whose main purpose is an
answer or reasoning (e.g. frustration) is not labelled, so affect rates read low. Accepted for
simplicity (decision 2026-10-13).

## 6. Quality gates, applied identically to every code

The same checks apply to all 57 codes, whatever their source:

1. **Positives before distillation.** Each code needs ≥ 300 positive turns in the 122B labels.
   Any code short of that gets the same **enrichment pass**: extra sessions whose candidate turns
   are pre-selected (by an existing child or a cheap retrieval), annotated with the unchanged 122B
   prompt. Enrichment changes which turns are labelled, never the label, and prevalence is
   reported on the random sample only.
2. **Distillation gate**, on held-out 122B-labelled turns: AUROC ≥ 0.75, per-turn SD of P ≥ 0.01,
   and some turns with P > 0.5. Class-weighted heads for every code under 5% prevalence. A code
   that fails is **reported**, then merged into its parent or left out; never shipped as a
   constant column.
3. **Every passing code gets its own column**: no importance-based pruning, so a code with low
   attribution is still in the model and can be reported as such.
4. **Per-code human agreement on the pilot**, not only an overall κ, since an overall κ
   can hide a code that never fires.

