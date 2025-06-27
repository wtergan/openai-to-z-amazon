# Results

This project should be read as an archaeological lead-generation workflow, not as a claim of confirmed discovery.

The most useful preserved outputs are the Checkpoint 2 ranked candidate files:

- `CHECKPOINT_2/test-run_top5.json`
- `CHECKPOINT_2/test-run_top5_llm.json`

Together, they show how the pipeline moved from multi-source geospatial features to ranked candidate cells and LLM-assisted interpretation.

## Checkpoint 2 output summary

The preserved Checkpoint 2 run documents a feature-engineering pass that produced:

- **3,407 H3 cells**;
- from **20,708 GEDI shots**;
- with **3,000 SRTM points**;
- followed by anomaly ranking and top-N candidate selection.

The ranked output file contains five candidate cells with full feature vectors. The LLM output file adds regional context and per-cell assessment.

## Preserved top-5 candidate assessment

In the preserved top-5 run:

- one candidate cell, `898aa919897ffff`, was rated `medium` potential with verification priority `3`;
- four other top-ranked cells were rated `low`;
- the `medium` candidate was notable because recent deforestation could expose otherwise hidden surface features;
- the regional assessment described dense, largely undisturbed riverine rainforest where subtle topography, paleochannels, and localized vegetation anomalies could be worth further review.

This is a modest but useful outcome: a structured shortlist of leads rather than a dramatic conclusion.

## What the ranking means

The anomaly pipeline combines multiple kinds of signals:

- terrain and elevation context;
- canopy or vegetation structure;
- disturbance / deforestation context;
- per-cell feature aggregation;
- weighted heuristic scores;
- Isolation Forest anomaly scores;
- LLM-assisted interpretation of ranked cells.

A high or medium-ranked cell should be interpreted as:

> a candidate location worth follow-up review under the assumptions of this workflow.

It should not be interpreted as:

> a confirmed archaeological site.

## Why the preserved outputs matter

The preserved outputs are important because they allow the project to be reviewed without requiring a full rerun of live geospatial services.

They show that the repo reached the point of:

1. collecting and caching geospatial data;
2. aggregating data into spatial cells;
3. producing engineered feature vectors;
4. ranking candidates;
5. using an LLM to explain and triage results.

For a portfolio/recruiter review, that is the core technical story.

## Follow-up work

If this project continued, the next result-focused improvements would be:

- produce a stable, documented offline demo from cached/mock data;
- add maps or static figures for the preserved top candidates;
- export a concise table of top-N candidate cells and key features;
- separate exploratory notebook outputs from curated documentation assets;
- add stronger validation against known archaeological site distributions where ethically and legally appropriate.
