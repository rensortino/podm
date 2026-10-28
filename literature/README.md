# Literature

The PDFs are git-ignored; `papers.tsv` is the index. To get every listed paper:

```bash
./literature/fetch.sh
```

To add a paper, append a tab-separated line `arxiv_id  filename  title (modules)` to `papers.tsv` and run the script. Non-arXiv papers: drop the PDF here and add a line with `-` as the id.

| Paper | Used in |
|---|---|
| [The Principles of Diffusion Models](https://arxiv.org/abs/2510.21890) — Lai, Song, Kim, Mitsufuji, Ermon (2025) | the main text for every module; page numbers in the course refer to this PDF |
