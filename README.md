# Where does a caption's humour come from?
*Words, pictures and ideas in the New Yorker Caption Contest*

## Abstract
Every week, thousands of readers try to caption the same New Yorker cartoon, and other readers rate the entries as unfunny, somewhat funny or funny. Across 365 crowdsourced contests this gives 2.2M captions and 270M ratings, a rare large-scale record of what an audience finds funny. We use it to ask where the funniness of a caption comes from. Is it in the **words**? Recycled captions such as "It's a living!" let us watch the same text succeed under one cartoon and fail under another. Is it in the **idea**? Hundreds of people often have the same joke, so the winner may be the most original or simply the best-worded version of a common thought. Finally, if we can model what this audience rewards, can **AI** write captions that the audience would rate highly, and would our judge be fooled by them? Our story moves from text, to picture, to idea, to machine.

## Research questions
1. **Same words, different joke.** When identical captions appear under different cartoons, how much of their score travels with the text, and how much depends on the picture (its "fit")? Which cartoon properties make a generic caption work?
2. **A common thought, perfectly expressed.** Do winning captions come from rare ideas or from popular ones? Within one shared idea, which wording differences go with higher ratings?
3. **Can AI please the crowd?** Do zero-shot captions from current vision-language models reach the level of human entries, as predicted by a judge trained on the human ratings and validated on unseen cartoons and a small blinded human comparison?

## Proposed additional datasets
No external dataset is required. We will **derive** three resources:
- **CLIP image embeddings** of all 385 cartoons (`open_clip`, < 1 MB). The provided metadata covers only 240 contests and stops at contest 763.
- **Sentence embeddings** of the ~2.1M unique normalised captions (`sentence-transformers` MiniLM, ≈ 1.6 GB as float16, computed once on Colab/cluster GPU and cached).
- **AI-generated captions**: 2–3 vision-language models × ~90 held-out cartoons × 10 captions (≈ 2,700 texts) through their APIs.
- *Optional:* contest dates from the nextml caption-contest-data repository, to replace contest-id bins with real time.

## Methods
All details, maths and pilot results are in [`results.ipynb`](results.ipynb).
- **Pipeline and cleaning.** Load, then flag internal inconsistencies (counts ≠ votes, mean ≠ counts, single-vote rows) and drop the pre-crowdsourcing contests (< 530). We keep 94.8% of rows.
- **Comparable scores.** Within-contest z-scores, plus empirical-Bayes shrinkage of each mean toward its contest mean using the reported standard error. Ratings come from a bandit (top captions get about 20× more votes), so we weight captions equally and bootstrap over contests.
- **RQ1.** Variance decomposition of recurring captions with a crossed random-effects model (caption effect, era effect, residual fit after subtracting known rating noise). The residual "fit" is then related to CLIP image–text similarity. *Pilot:* the same text has an ICC of 0.52–0.65 within a contest but only 0.23 across cartoons.
- **RQ2.** Per-contest clustering of sentence embeddings into joke ideas. Idea popularity (cluster size, entropy of cluster sizes) is related to scores, and best and worst wordings within a cluster are compared with paired tests. *Pilot:* captions sharing their idea with more entries score higher (Spearman 0.12), and top-10 captions are less often unique (26% vs 39%).
- **RQ3.** A judge (caption + image embeddings → z-score, or Bradley–Terry pairwise) trained with contests held out, evaluated by pairwise ranking accuracy (top-10 vs #1000–1009). *Pilot:* a text-only ridge model reaches 63.0% ± 1.4%, on par with crowd workers and GPT-4 in the dataset paper. The multimodal judge must beat this.

## Proposed timeline
| Week | Dates | Goal |
|---|---|---|
| 1 | 14–20 Nov | Compute and cache sentence + CLIP embeddings. Integrate feedback from P2 |
| 2 | 21–27 Nov | RQ1 mixed model. RQ2 clustering, tuned by hand on 5 contests |
| 3 | 28 Nov – 4 Dec | RQ2 analyses. RQ3 caption generation and multimodal judge |
| 4 | 5–11 Dec | RQ3 evaluation + blinded human check. Freeze results |
| 5 | 12–18 Dec | Data story website, figures, README cleanup. **P3 deadline 18 Dec** |

## Organization within the team
*(fill in names)*
- **Member A:** data pipeline, embeddings, repository hygiene
- **Member B:** RQ1 (mixed model, CLIP fit analysis)
- **Member C:** RQ2 (clustering, wording analysis)
- **Member D:** RQ3 (generation, judge, human study)
- **Everyone:** data story writing and visualisations in week 5

## Questions for TAs
- For RQ3, is a small blinded human evaluation (e.g. ~10 raters among classmates) acceptable, or should we stay with model-based evaluation only?
- Is it acceptable to compute embeddings on Colab/the cluster and commit only the cached arrays (or a download script)?

## Repository structure
```
├── results.ipynb          # main notebook (P2 results)
├── src/
│   ├── data_loading.py    # CSV/JSON -> tidy tables, parquet cache
│   ├── quality.py         # consistency flags, cleaning report
│   ├── scoring.py         # z-scores, empirical-Bayes means, ICC + bootstrap
│   ├── text.py            # normalisation, text features, TF-IDF idea similarity
│   └── judge.py           # text-only baseline judge, pairwise accuracy
├── data/                  # (git-ignored) caches and processed tables
└── requirements.txt
```
Put the dataset folder `newyorker_caption_contest/` next to this repository (or inside it), then run `results.ipynb`.

*Use of LLMs:* an LLM assistant (Claude) helped draft code and text; the team ran, checked and interpreted all analyses.
