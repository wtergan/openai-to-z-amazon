# Source Library

This project was grounded in challenge-specific materials, Amazonian archaeology, remote-sensing methods, public geospatial datasets, and ethics / interpretation caveats. This page is the curated entry point; the broader Vault-derived index is in [`comprehensive-source-index.md`](comprehensive-source-index.md).

## How to use this library

- Start with **Core sources** to understand why the repo frames the challenge as geospatial prioritization rather than confirmed discovery.
- Use **Supporting source clusters** to validate specific implementation choices: Sentinel-2 imagery, LiDAR/DEM terrain context, SRTM, GEDI-style canopy structure, forest-change layers, and LLM-assisted triage.
- Use the comprehensive index for the full public-facing source trail extracted from the Vault corpus.

## Core sources

### OpenAI to Z Challenge | OpenAI

- URL: <https://openai.com/openai-to-z-challenge/>
- Role in this project: canonical challenge framing and public problem statement
- Why it matters: This is the canonical public framing of the OpenAI to Z Challenge.

### Checkpoints: OpenAI to Z Challenge

- URL: <https://cdn.openai.com/pdf/a9455c3b-c6e1-49cf-a5cc-c40ed07c0b9f/checkpoints-openai-to-z-challenge.pdf>
- Role in this project: official checkpoint structure and expected deliverables
- Why it matters: This PDF is the clearest surviving artifact for reconstructing what a serious OpenAI to Z submission was supposed to look like.

### Starter Pack: OpenAI to Z Challenge

- URL: <https://cdn.openai.com/pdf/a9455c3b-c6e1-49cf-a5cc-c40ed07c0b9f/starter-pack-openai-to-z-challenge.pdf>
- Role in this project: official starter framing, data suggestions, and evidence-fusion orientation
- Why it matters: This PDF is one of the highest-value reconstruction sources in the whole bookmark set because it reveals the intended shape of the OpenAI to Z workflow.

### OpenAI to Z Challenge Kaggle Starter Materials

- URL: <https://www.kaggle.com/competitions/openai-to-z-challenge/discussion/579189>
- Role in this project: public challenge ecosystem and starter-material discussion
- Why it matters: This note stands in for the challenge's starter-materials thread.

### 🛰️Search the Amazon with Remote Sensing and AI🤖

- URL: <https://www.kaggle.com/code/fnands/search-the-amazon-with-remote-sensing-and-ai>
- Role in this project: early remote-sensing plus AI workflow pattern
- Why it matters: This Kaggle notebook appears to be an early OpenAI-to-Z challenge notebook focused on remote sensing plus model-assisted interpretation.

### [OpenAI to Z Challenge] Checkpoint 2

- URL: <https://www.kaggle.com/code/ndy001/openai-to-z-challenge-checkpoint-2#Checkpoint-2>
- Role in this project: comparison point for the anomaly-ranking phase
- Why it matters: This Kaggle notebook is clearly related to the challenge's second-stage or "early explorer" work, but direct extraction from Kaggle failed because the page crashed during access.

### Predicting the geographic distribution of ancient Amazonian archaeological sites with machine learning - PMC

- URL: <https://pmc.ncbi.nlm.nih.gov/articles/PMC10069417/>
- Role in this project: methodological precedent for predictive archaeological site modeling
- Why it matters: This 2023 PeerJ paper is one of the clearest methodological precedents for the project.

### More than 10,000 pre-Columbian earthworks are still hidden throughout Amazonia | Science

- URL: <https://www.science.org/doi/10.1126/science.ade2541>
- Role in this project: quantitative support for the plausibility of undiscovered Amazonian earthworks
- Why it matters: quantitative support for the plausibility of undiscovered Amazonian earthworks

### Pre-Columbian earth-builders settled along the entire southern rim of the Amazon | Nature Communications

- URL: <https://www.nature.com/articles/s41467-018-03510-7>
- Role in this project: regional archaeology support for settlement-pattern reasoning
- Why it matters: This 2018 Nature Communications paper is one of the most important archaeology sources for the repo because it links newly documented Upper Tapajós Basin earthworks to a much broader southern rim of Amazonia pattern.

### Lidar reveals pre-Hispanic low-density urbanism in the Bolivian Amazon - PMC

- URL: <https://pmc.ncbi.nlm.nih.gov/articles/PMC9177426/>
- Role in this project: remote-sensing precedent for seeing archaeological structure under canopy
- Why it matters: This 2022 Nature paper is one of the strongest domain sources in the vault because it shows what high-quality airborne lidar can actually reveal in Amazonia when coverage and archaeological context are good enough.

### Historical Human Footprint on Modern Tree Species Composition in the Purus-Madeira Interfluve, Central Amazonia | PLOS One

- URL: <https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0048559>
- Role in this project: evidence for long-lived environmental signatures of past human activity
- Why it matters: This PLOS ONE paper provides ecological evidence that supposedly "primary" forests in the Purus-Madeira interfluve still bear traces of past human management.

### Ethics in Archaeological Lidar | Journal of Computer Applications in Archaeology

- URL: <https://journal.caa-international.org/articles/10.5334/jcaa.48>
- Role in this project: ethical caveats for handling candidate locations and avoiding overclaiming
- Why it matters: Damian Evans, Sarah Klassen, and Anna Cohen argue that archaeological lidar has become powerful enough that method discussions are no longer sufficient on their own; projects also need explicit ethical choices around access, stakeholder involvement, and public communication.

## Supporting source clusters

These grouped references expand the research trail without making every item a core methodological citation. They are useful for future repo work, blog writing, or independent validation.

### Challenge context and implementation examples

- [Amazon AI Discovery:Sentinel-2, GEE, DEM & Open AI](https://www.kaggle.com/code/superdragon/amazon-ai-discovery-sentinel-2-gee-dem-open-ai#9.-Execute-Checkpoint-1:-GEE-Image-+-OpenAI-Vision-Analysis)
- [Baseline - Team: KOFUN to Z](https://www.kaggle.com/code/isakatsuyoshi/baseline-team-kofun-to-z#1.-Loads-two-independent-public-sources)
- [Checkpoint_1: Loading LiDAR Dataset + GPT Analysis](https://www.kaggle.com/code/utkarshranjan007/checkpoint-1-loading-lidar-dataset-gpt-analysis)
- [I Follow Rivers](https://www.kaggle.com/code/fafa92/i-follow-rivers)
- [I Follow Rivers](https://www.kaggle.com/code/fafa92/i-follow-rivers/notebook)
- [Naked Eyes - 100+ Potential New Sites](https://www.kaggle.com/code/fafa92/naked-eyes-100-potential-new-sites/notebook)
- [OpenAI to Z Challenge | Kaggle](https://www.kaggle.com/competitions/openai-to-z-challenge/discussion/579267)
- [OpenAI to Z Challenge | Kaggle](https://www.kaggle.com/competitions/openai-to-z-challenge/overview)

### Amazonian archaeology, history, and settlement context

- [(PDF) Two thousand years of garden urbanism in the Upper Amazon](https://www.researchgate.net/publication/377333495_Two_thousand_years_of_garden_urbanism_in_the_Upper_Amazon)
- [Chapter-8: Peoples of the Amazon before European Colonization.pdf](https://www.theamazonwewant.org/wp-content/uploads/2022/05/Chapter-8-Bound-May-9.pdf)
- [Cristóbal de Acuña, New Discovery of the Great River of the Amazon. Study, edition and notes by Ignacio Arellano, José M. Díez Borque and Gonzalo Santonja. Madrid/Frankfurt am Main, Iberoamericana/Vervuert, 2009. 184 p.](https://journals.openedition.org/criticon/15929?lang=en)
- [Gerhard Bechtold: Terra Preta](https://www.gerhardbechtold.com/TP/gbtp.html)
- [Lost Garden Cities: Pre-Columbian Life in the Amazon | Scientific American](https://www.scientificamerican.com/article/lost-cities-of-the-amazon/)
- [Map of the Brazilian Amazon region with known Terra Preta sites (open... | Download Scientific Diagram](https://www.researchgate.net/figure/Map-of-the-Brazilian-Amazon-region-with-known-Terra-Preta-sites-open-boxes-and_fig1_225201533)
- [Nuevo descubrimiento del gran rio de las Amazonas - Cristóbal de Acuña - Google Books](https://books.google.com/books?id=R28BAAAAQAAJ&printsec=frontcover&source=gbs_ge_summary_r&cad=0#v=onepage&q&f=false)
- [Pre-Columbian Amazon supported millions of people](https://news.mongabay.com/2005/10/pre-columbian-amazon-supported-millions-of-people/#:~:text=The%20idea%20that%20the%20Amazon,%E2%80%9D)
- [Relation historique et geographique, de la grande riviere des Amazones dans l’Amerique Par le Comte de Pagan. Extraicte de divers Autheurs, & reduitte en meilleure forme. Avec la Carte d’icelle Riviere, & de](https://www.foldvaribooks.com/pages/books/2396/blaise-francois-de-pagan/relation-historique-et-geographique-de-la-grande-riviere-des-amazones-dans-l-amerique-par-le-comte)
- [The Lost City of Z: A Journey into the History, Discoveries and Mythology of a Legend — Louis Wolf](https://louiswolf.com/english/2023/6/3/the-lost-city-of-z-a-journey-into-the-history-discoveries-and-mythology-of-a-legend)
- [The Pre-Columbian Peopling and Population Dispersals of South America | Journal of Archaeological Research](https://link.springer.com/article/10.1007/s10814-020-09146-w)
- [Uncovering Amazonia | Open Rivers Journal](https://openrivers.lib.umn.edu/article/uncovering-amazonia/#:~:text=Later%2C%20Carvajal%20talks%20about%20traveling,gleaming%20white%20in%20the%20distance)

### LiDAR, DEM, terrain, and relief visualization

- [Airborne-laser-scanning-raster-data-visualization-A-Guide-to-Good-Practice.pdf](https://www.researchgate.net/profile/Ziga-Kokalj/publication/314976947_Airborne_laser_scanning_raster_data_visualization_A_Guide_to_Good_Practice/links/58c7f251aca2723ab1653a86/Airborne-laser-scanning-raster-data-visualization-A-Guide-to-Good-Practice.pdf)
- [CMS: LiDAR Data for Forested Areas in Paragominas, Para, Brazil, 2012-2014, https://doi.org/10.3334/ORNLDAAC/1302](https://daac.ornl.gov/cgi-bin/dsviewer.pl?ds_id=1302)
- [Hillshade - Spatial Analysis | Atlas](https://atlas.co/spatial-analysis/hillshade/#:~:text=Hillshade%20,by%20highlighting%20the%20play)
- [Impact of LiDAR pulse density on forest fuels metrics derived using LadderFuelsR - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S157495412500144X)
- [L1A - Discrete airborne LiDAR transects collected by EBA in the Brazilian Amazon (Acre e Rondônia)](https://zenodo.org/records/7689909)
- [L2C - Canopy height models across the Brazilian Amazon](https://zenodo.org/records/7104044)
- [LiDAR and DTM Data from Forested Land Near Manaus, Amazonas, Brazil, 2008](https://daac.ornl.gov/VEGETATION/guides/Forested_Areas_Amazonas_Brazil.html)
- [LiDAR and DTM Data from Forested Land Near Manaus, Amazonas, Brazil, 2008, https://doi.org/10.3334/ORNLDAAC/1515](https://daac.ornl.gov/cgi-bin/dsviewer.pl?ds_id=1515)
- [LiDAR and DTM Data from Tapajos National Forest in Para, Brazil, 2008](https://daac.ornl.gov/VEGETATION/guides/Forested_Areas_Para_Brazil.html)
- [LiDAR Surveys over Selected Forest Research Sites, Brazilian Amazon, 2008-2018, https://doi.org/10.3334/ORNLDAAC/1644](https://daac.ornl.gov/cgi-bin/dsviewer.pl?ds_id=1644)
- [OpenTopography - Shuttle Radar Topography Mission (SRTM) Global](https://portal.opentopography.org/datasetMetadata?otCollectionID=OT.042013.4326.1)
- [Visualizing Topography by Openness:A New Application of Image Processing to Digital Elevation Models](https://www.asprs.org/wp-content/uploads/pers/2002journal/march/2002_mar_257-265.pdf)

### Satellite imagery, vegetation indices, and geospatial tooling

- [gdalcubes 0.7.0 - 2. Data cubes from Sentinel-2 data in the cloud](https://gdalcubes.github.io/source/tutorials/vignettes/gc02_AWS_Sentinel2.html)
- [Hansen Global Forest Change v1.11 (2000-2023) | Earth Engine Data Catalog | Google for Developers](https://developers.google.com/earth-engine/datasets/catalog/UMD_hansen_global_forest_change_2023_v1_11)
- [Landsat & Sentinel-2 Data on AWS | Geospatial Python](https://www.matecdev.com/posts/landsat-sentinel-aws-s3-python.html)
- [S2 Applications](https://sentiwiki.copernicus.eu/web/s2-applications)
- [Seeing Earth from Space — from Raw Satellite Data to Beautiful High-resolution Images | by Antti Lipponen | Medium](https://medium.com/@anttilip/seeing-earth-from-space-from-raw-satellite-data-to-beautiful-high-resolution-images-feb522adfa3f)
- [Sentinel2 images exploration and processing with Python and Rasterio - Tutorial — Hatari Labs](https://hatarilabs.com/ih-en/sentinel2-images-explotarion-and-processing-with-python-and-rasterio)
- [VEGETATION ANALYSIS : NDVI CALCULATION ON GOOGLE EARTH ENGINE | by PETER NDIRITU THUKU | Medium](https://medium.com/@thukupeter487/vegetation-analysis-ndvi-calculation-on-google-earth-engine-dcce6d951220)

### Ethics, ecology, and interpretive caveats

- [Ancient Amazonian populations left lasting impacts on forest structure - Palace - 2017 - Ecosphere - Wiley Online Library](https://esajournals.onlinelibrary.wiley.com/doi/10.1002/ecs2.2035#:~:text=Ancient%20Amazonian%20populations%20left%20lasting,were%20more%20pronounced%20in)
- [Bamboo-Dominated Forests of the Southwest Amazon: Detection, Spatial Extent, Life Cycle Length and Flowering Waves - PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC3554598/)
- [Climate Change and Amazon Floods (brainstorming)](https://www.kaggle.com/code/woosungyoon/climate-change-and-amazon-floods-brainstorming)
- [Drivers of geophagy of large-bodied amazonian herbivorous and frugivorous mammals | Scientific Reports](https://www.nature.com/articles/s41598-024-80237-0)
- [How human activity has shaped Brazil Nut forests’ past and future](https://www.mpg.de/24062520/how-human-activity-has-shaped-brazil-nut-forests-past-and-future)
- [NOAA/WDS Paleoclimatology - Cueva del Tigre Perdido, Peru Stalagmite Fluid Inclusion Isotope Data](https://www.ncei.noaa.gov/access/metadata/landing-page/bin/iso?id=noaa-cave-9790)
- [Uncovering spatial patterns in the natural and human history of Brazil nut (Bertholletia excelsa) across the Amazon Basin](https://cgspace.cgiar.org/items/ec098f68-493c-4c67-82b0-ad775a7eb068)

### Machine learning and future-method references

- [(PDF) Objective comparison of relief visualization techniques with deep CNN for archaeology](https://www.researchgate.net/publication/351426881_Objective_comparison_of_relief_visualization_techniques_with_deep_CNN_for_archaeology)
- [(PDF) Why Not a Single Image? Combining Visualizations to Facilitate Fieldwork and On-Screen Mapping](https://www.researchgate.net/publication/332075411_Why_Not_a_Single_Image_Combining_Visualizations_to_Facilitate_Fieldwork_and_On-Screen_Mapping)
- [https://ampcode.com/how-to-build-an-agent](https://ampcode.com/notes/how-to-build-an-agent)
- [Machine learning applications in archaeological practices: a review](https://arxiv.org/pdf/2501.03840v1)
- [Major Tom Embedding Filter](https://www.kaggle.com/code/fnands/major-tom-embedding-filter)
- [Transfer Learning of Semantic Seg. Methods for Identifying Buried Archaeo Structures on LiDAR Data](https://arxiv.org/pdf/2307.03512)

## Data and method references used by the repo

| Repo concern | Useful source cluster |
| --- | --- |
| Challenge framing and checkpoints | OpenAI challenge page, checkpoint PDF, starter pack, Kaggle overview/discussions, starter notebooks |
| Predictive archaeology framing | Predictive site-distribution modeling, hidden-earthworks estimates, southern-rim earth-builder studies, Bolivian lidar urbanism |
| Sentinel-2 and spectral imagery | Sentinel-2 applications, NDVI/GEE examples, Landsat/Sentinel cloud access, Hansen forest-change data, Rasterio/gdalcubes workflows |
| Terrain / DEM / LiDAR context | OpenTopography SRTM, ORNL DAAC Amazon lidar inventories, hillshade / relief visualization references, canopy-height products |
| Ethics and caveats | Archaeological lidar ethics, anthropogenic forest signatures, ecological confounders, disturbance/false-positive context |

## Comprehensive source index

For the expanded bibliography-style source list, see [`docs/research/comprehensive-source-index.md`](comprehensive-source-index.md). It includes 100+ public source URLs extracted from the Vault corpus, grouped by role, with ingestion status and triage tier where available.

## How these sources shaped the repo

The design pattern that emerged from this corpus was:

1. use archaeological literature to define plausible search signals;
2. use public geospatial datasets to approximate those signals;
3. aggregate observations into spatial cells;
4. rank cells with anomaly detection and heuristic scoring;
5. use LLMs to explain, compare, and triage candidate leads;
6. document uncertainty rather than overclaiming results.
