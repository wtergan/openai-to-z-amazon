# Source Library

This project was grounded in both challenge-specific material and archaeological / remote-sensing literature. The sources below are the most relevant references for understanding the project framing, pipeline design, and limitations.

## Challenge and implementation context

### OpenAI to Z Challenge

- URL: <https://openai.com/openai-to-z-challenge/>
- Role in this project: canonical challenge framing.
- Why it matters: establishes the project goal, timeline, and expected use of open data and AI systems for Amazon archaeological search.

### Official checkpoint document

- URL: <https://cdn.openai.com/pdf/a9455c3b-c6e1-49cf-a5cc-c40ed07c0b9f/checkpoints-openai-to-z-challenge.pdf>
- Role in this project: checkpoint structure and submission expectations.
- Why it matters: explains why the repo is organized around Checkpoint 1 and Checkpoint 2 rather than as a single polished package.

### Kaggle starter materials discussion

- URL: <https://www.kaggle.com/competitions/openai-to-z-challenge/discussion/579189>
- Role in this project: practical challenge ecosystem context.
- Why it matters: helps connect the repo to the public starter materials, data suggestions, and challenge workflow.

### Search the Amazon with Remote Sensing and AI

- URL: <https://www.kaggle.com/code/fnands/search-the-amazon-with-remote-sensing-and-ai>
- Role in this project: early remote-sensing + model-assistance pattern.
- Why it matters: provides context for combining satellite/terrain data with AI-assisted interpretation.

### OpenAI to Z Challenge Checkpoint 2 notebook

- URL: <https://www.kaggle.com/code/ndy001/openai-to-z-challenge-checkpoint-2#Checkpoint-2---An-Early-Explorer>
- Role in this project: Checkpoint 2 implementation context.
- Why it matters: aligns with the repo's move from exploratory feature extraction into anomaly ranking and candidate assessment.

## Archaeological and methodological grounding

### Predicting the geographic distribution of ancient Amazonian archaeological sites with machine learning

- URL: <https://pmc.ncbi.nlm.nih.gov/articles/PMC10069417/>
- Role in this project: methodological precedent for predictive archaeological site modeling.
- Why it matters: supports the idea that archaeological search can be framed as probabilistic geospatial ranking rather than certainty-based discovery.

### More than 10,000 pre-Columbian earthworks are still hidden throughout Amazonia

- Role in this project: plausibility and scale of undiscovered archaeological features.
- Why it matters: reinforces that the Amazon can contain many hidden or under-documented archaeological features, making remote-sensing triage a meaningful problem.

### Pre-Columbian earth-builders settled along the entire southern rim of Amazonia

- URL: <https://www.nature.com/articles/s41467-018-03510-7>
- Role in this project: regional archaeological context.
- Why it matters: broadens the search framing beyond isolated known sites and supports settlement-pattern reasoning.

### Lidar reveals pre-Hispanic low-density urbanism in the Bolivian Amazon

- URL: <https://pmc.ncbi.nlm.nih.gov/articles/PMC9177426/>
- Role in this project: high-quality remote-sensing archaeology precedent.
- Why it matters: demonstrates how lidar and terrain analysis can reveal settlement patterns that are difficult to observe under vegetation.

### Historical human footprint on modern tree species composition in the Purus-Madeira interfluve, central Amazonia

- URL: <https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0048559>
- Role in this project: environmental signature context.
- Why it matters: supports the idea that past human activity can leave indirect signals in forest composition and landscape structure.

### Ethics in Archaeological Lidar

- URL: <https://journal.caa-international.org/articles/10.5334/jcaa.48>
- Role in this project: ethical framing.
- Why it matters: reinforces that candidate locations should be handled carefully and that remote-sensing outputs should not be overclaimed or exposed irresponsibly.

## Data and remote-sensing context

### OpenTopography / SRTM terrain data

- Role in this project: terrain and elevation context.
- Why it matters: SRTM and DEM-derived products provide terrain structure that can help identify anomalies, paleochannels, mounds, or other candidate features.

### Sentinel-2 imagery

- Role in this project: multispectral surface context.
- Why it matters: vegetation indices and spectral signals can provide indirect evidence of surface disturbance, vegetation differences, or land-cover transitions.

### GEDI canopy metrics

- Role in this project: canopy structure context.
- Why it matters: canopy height and structure can help characterize forest conditions and possible anomalies when aggregated spatially.

### PRODES disturbance context

- Role in this project: deforestation / disturbance layer.
- Why it matters: recently disturbed areas may expose surface features, but disturbance can also create false positives and must be interpreted carefully.

## How these sources shaped the repo

The core design pattern that emerged from these sources was:

1. use archaeological literature to define plausible search signals;
2. use public geospatial datasets to approximate those signals;
3. aggregate observations into spatial cells;
4. rank cells with anomaly detection and heuristic scoring;
5. use LLMs to explain, compare, and triage candidate leads;
6. document uncertainty rather than overclaiming results.

## Sources to review next

If this source library is expanded, the next useful additions would be:

- a short annotated bibliography for each core paper;
- notes on known-site validation datasets and ethical constraints;
- a false-positive guide for ecological and geomorphological confounders;
- a comparison table of lidar, SRTM, Sentinel-2, GEDI, and PRODES roles in the pipeline.
