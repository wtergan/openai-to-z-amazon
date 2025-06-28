# Reproducibility

This repository preserves a challenge-time geospatial AI workflow. It is reproducible in parts, but not every path is currently one-command reproducible because the project depends on live APIs, Google Earth Engine authentication, and external datasets.

## Reproducibility summary

| Area | Status | Notes |
| --- | --- | --- |
| Repository inspection | Works | Code, notebooks, cached files, and preserved outputs are available. |
| Checkpoint notebooks | Preserved | Useful as historical artifacts; full reruns require dependencies and credentials. |
| Checkpoint 1 LiDAR path | Most likely to rerun | Requires OpenTopography API access. |
| Checkpoint 1 Sentinel-2 path | Credentialed | Requires Google Earth Engine auth. |
| Checkpoint 2 full pipeline | Credentialed / drift-prone | Requires Earth Engine, dependencies, and live services. |
| Checkpoint 2 preserved outputs | Works | JSON artifacts can be inspected without rerunning. |
| Checkpoint 2 tests | Needs cleanup | Some tests currently mix offline, integration, and stale assumptions. |

## Preserved artifacts

The safest way to review the project today is to inspect preserved outputs:

- `CHECKPOINT_2/test-run_top5.json`
- `CHECKPOINT_2/test-run_top5_llm.json`
- `CHECKPOINT_1/Checkpoint_1.ipynb`
- `CHECKPOINT_2/Checkpoint_2.ipynb`

These artifacts preserve the challenge work even when external services drift.

## What should work after setup

After installing dependencies and providing credentials, the following paths are intended to be runnable:

```bash
cd CHECKPOINT_1
python console_output.py
```

and, for Checkpoint 2:

```bash
cd CHECKPOINT_2
python console_output.py --bbox=-59.9,-5.2,-59.8,-5.1 --year 2022 --top_n 5
```

The Checkpoint 2 path should be treated as a credentialed geospatial workflow. It may fail if Earth Engine authentication, dataset endpoints, or model/provider APIs have changed.

## Offline and integration test boundary

The test suite now separates standard-library smoke checks from dependency-heavy checks.

Default local test command:

```bash
PYTHONPATH=CHECKPOINT_2 pytest CHECKPOINT_2/tests -q
```

This runs preserved-artifact smoke tests and skips live-service tests unless explicitly enabled.

Optional integration flags:

```bash
RUN_GEE_INTEGRATION_TESTS=1 PYTHONPATH=CHECKPOINT_2 pytest CHECKPOINT_2/tests/test_pipeline.py CHECKPOINT_2/tests/test_srtm.py -q
RUN_API_INTEGRATION_TESTS=1 PYTHONPATH=CHECKPOINT_2 pytest CHECKPOINT_2/tests/test_with_api.py -q
```

The focused offline/mock Checkpoint 2 target remains:

```bash
PYTHONPATH=CHECKPOINT_2 pytest CHECKPOINT_2/tests/test_pipeline.py::test_mock_data -q
```

That target requires the geospatial dependencies from `CHECKPOINT_2/requirements.txt`, but it does not require live API calls.

## External services and drift

This project depends on several services that can change independently of the repository:

- Google Earth Engine authentication and dataset availability;
- OpenTopography API access and rate limits;
- OpenAI model availability and response behavior;
- OpenRouter routing/model availability;
- public geospatial datasets and schemas.

Because of that, the repository distinguishes between:

1. **historical artifacts** — notebooks, cached data, and preserved outputs;
2. **currently inspectable code** — modularized source files and docs;
3. **credentialed reruns** — paths requiring live services and valid API keys.

## Interpretation limits

The project produces ranked archaeological leads, not proof of discoveries. LLM outputs are used for interpretation and triage. They should be read as structured reasoning over geospatial context, not as archaeological validation.

The correct standard for this repo is therefore:

- preserve the evidence trail;
- document assumptions and limitations;
- make the runnable pieces clearer over time;
- avoid overstating candidate rankings as confirmed sites.
