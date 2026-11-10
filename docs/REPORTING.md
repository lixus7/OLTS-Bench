**English** | [简体中文](REPORTING.zh-CN.md)

# How to write a report like ours

Examples (in Chinese):
- [the full brief, 29 pages](../reports/full_study/BUA_brief_full_zh.pdf);
- [the two-page summary](../reports/full_study/BUA_brief_onepage_zh.pdf);
- [the auto-generated draft from the ETTh1 quick run](../reports/example_etth1_quick/), in Chinese and English.

Skim the full brief first, then read the rules below.

## 1. Two documents

- **A summary (1–2 pages)** for busy readers. It has, in order:
  - the question;
  - a one-line answer;
  - the two main tables as heat maps;
  - a few key points;
  - open questions for discussion.

  A second page can hold the supplementary experiments.
- **The full version** for readers who will check and ask questions. Every definition, example, table and limitation is in it.

## 2. Structure of the full version

| Part | Contents |
|---|---|
| Opening | The research question (one paragraph); a one-line verdict; 4–5 key points, each with a number and a table reference; open questions; a status line (what is new, what is re-analysed, results as of which date, what this version adds) |
| Section 1 Design | One diagram: what each model sees at time t. What each comparison answers |
| Section 2 Setup | Data, splits, models, methods, training budget, causal rule, scoring; the definition of "a pair" with one concrete example |
| Results sections | One question per section: first "how to read", then the table or figure, then the key points. Every table and figure is numbered |
| Supplementary experiments | Each subsection has the same parts: why, how, the pre-registered reading, how to read, table, results, guesses, caveats |
| Conclusions | Two columns: what we can say (within this scope) and what we cannot say yet (not supported, out of scope, or only a guess) |
| Discussion | Questions to decide together, each with the evidence so far |
| Appendix | Background study, changes of scope, data processing, glossary |

## 3. Writing rules

Each of these comes from a mistake we made once.

1. **Answer first, evidence second.** The first sentence answers the question; the sections behind it support it with tables.
2. **Every percentage says who is compared with whom, and whether + is better or worse.**
   - Write "the online error is 7.4% lower than the frozen model's", not just "+7.4%".
   - Turn large negatives into multiples: −250% means "3.5 times the frozen error".
   - When there are two references, name the reference in every sentence ("against the tuned frozen model", "against the main-table frozen model").
3. **Define every term at first use, with one concrete example.** This covers terms such as gain, pair, multiplier and warm-up. Translate internal shorthand: "multiplier 1" becomes "not reduced, as in the main tables".
4. **Report medians together with counts.** Write "k/n": the online model was better in k of n pairs. Mention diverging runs separately; a mean is at most a warning sign.
5. **Number every section, table and figure, and refer to them by number** ("see Table 7", not "the table above"). When revising, keep numbers that readers have already seen; add new tables at the end, or say in the status line what moved.
6. **Name things; avoid vague phrases.** Not "the wider datasets" but "Weather (21 variables), Solar (137), PEMS08 (170), Electricity (321), PEMS-BAY (325)".
7. **Keep what we can say apart from what we cannot say yet.** Label guesses as guesses and say how to test them; say so when a statement rests on a single run.
8. **Write the reading rule down before the run.**
   - Afterwards, report plainly whether it held; never move the line.
   - Mark comparisons added after seeing results as "not in the pre-registered reading".
   - Where the rule was ambiguous (for example, which reference it meant), say which reading you used and what the other reading gives.
9. **State the scope:** which datasets, windows, horizon and seeds. A conclusion holds only inside that scope.
10. **Do not compare absolute errors across splits.** O and T differ in TEST period and training size; compare online against frozen within a split only.
11. **When new results change earlier text, update the earlier text.**
    - For example, "planned" becomes "see Section 8.7";
    - correct earlier mistakes;
    - do not break anything else on the way.
12. **Keep the tone neutral.** No hype, and no words like "significant" or "dramatic" without a number behind them. Notes for supervisors and colleagues need no politeness padding; mark disagreements "to discuss".
13. **Reports for classmates are about the science only.** Leave out GPU hours, job ids and cluster names, and follow each result with an explanation or a guess.
14. **Report experiment status truthfully.**
    - "Scheduled" only after it was really submitted;
    - "queued" is not "running";
    - "finished" is not "checked".
15. **Never type a number by hand.** A script generates every number in the text from the result files and asserts each one. When data change, the assertions tell you which sentence to fix.
16. **Before sending, have someone who did not do the work read it cold** (an AI reviewer works too). Ask them to look for:
    - comparisons without a stated reference;
    - contradictions;
    - stale statements;
    - undefined terms;
    - claims stronger than the data.

## 4. From draft to final

1. **Run the experiment.** `python -m olts.run <config>` writes:
   - `results/summary.json`;
   - `results/report_en.html` (the draft: every table, number and cell-by-cell sentence filled in).
2. **Check the numbers.** When you reproduce the study's settings, run `python scripts/compare_with_study.py runs/<name>`.
3. **Write the verdict and the explanations:** the key points, "can say / cannot say yet" and the guesses, each pointing to a table.
4. **Let code own the numbers.** Copy `olts/report.py` and add your paragraphs:
   - take each number from `summary.json` with an f-string;
   - add one `assert` for each number.
5. **Cold read.** Use the checklist below, or ask someone.
6. **Make the PDF.** Run `bash scripts/html_to_pdf.sh <report.html>`, then check the page breaks: no table split across pages, no heading at the bottom of a page.

## 5. Checklist before sending

- [ ] Is the one-line verdict directly supported by the main tables?
- [ ] Does every percentage say whom it compares and what its sign means?
- [ ] Does every table and figure have a number, a "how to read" and its counts?
- [ ] Is every term defined, with an example, at first use?
- [ ] Is the scope (datasets, windows, H, seeds) stated?
- [ ] Are guesses labelled, and "can say" kept apart from "cannot say yet"?
- [ ] Are there stale statements? Has anything "planned" already been done?
- [ ] Do all numbers come from scripts, with assertions passing?
- [ ] Are the PDF page breaks clean?
