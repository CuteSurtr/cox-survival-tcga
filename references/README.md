# References

Curated open-access literature for the mutation-signatures NMF project.

## Contents

- `queries.yaml` -- human-editable list of theses, papers, and API searches
- `fetch.py`     -- downloader + auto-generated bibliography builder
- `pdfs/`        -- downloaded PDFs (git-ignored; reproducible from fetch)
- `cache/`       -- cached API responses (git-ignored)
- `BIBLIOGRAPHY.md` -- generated index with source links + local file paths

## Usage

```bash
# install PyYAML and requests if needed (already in requirements.txt)
pip install pyyaml requests

# do a dry run first to see what will be fetched
python references/fetch.py --dry-run

# actually fetch everything marked fetch: true
python references/fetch.py

# only a single section
python references/fetch.py --only theses
python references/fetch.py --only searches
```

## Adding a new reference

Open `queries.yaml` and append a block under the right section:

```yaml
papers:
  - id: lastname_year_shortname
    author: "Lastname A, Lastname B"
    year: 2025
    title: "Paper title goes here"
    venue: "Journal name"
    pmcid: "PMC1234567"        # if open in Europe PMC
    doi:   "10.xxxx/yyyy"
    kind: pmc                   # pmc | pdf | landing_page
    fetch: true
    notes: >
      Why this paper is in our bibliography and what it contributes.
```

For theses, use the same schema under `theses:`. If you find a PhD thesis at a
university repository (Cambridge Apollo, MIT DSpace, Stanford SearchWorks,
etc.), link directly to the PDF URL.

## Starting points for expanding the thesis list

The Wellcome Sanger Institute thesis collection is a goldmine for
mutation-signatures-related PhDs:

- Cambridge Apollo -- Sanger Institute collection:
  https://www.repository.cam.ac.uk/collections/54880bf5-8828-4525-a20d-9c5a4d5c5cff

Other repositories worth browsing:

- MIT DSpace (computational biology): https://dspace.mit.edu/
- Stanford Digital Repository: https://purl.stanford.edu/
- DART-Europe (European open theses aggregator): https://www.dart-europe.org/
- British Library EThOS: https://ethos.bl.uk/
- OpenThesis: https://www.openthesis.org/
- ProQuest Open Access theses: https://www.proquest.com/openview

## Respect for publishers

The fetcher only downloads:

1. URLs explicitly marked open access in `queries.yaml`,
2. PMC open-access PDFs via the Europe PMC official endpoint,
3. arXiv PDFs,
4. bioRxiv / medRxiv preprint PDFs.

It does **not** try to bypass paywalls, proxy through institutional logins, or
harvest from Sci-Hub / equivalent. If a paper you want is paywalled, add the
`landing_page` kind and `fetch: false` so it appears in the bibliography
without being downloaded.
