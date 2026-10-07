# IspovestiBot

Generates short confessions in the style of [ispovesti.com](https://ispovesti.com), conditioned on how the community would react to them.

This is a bachelor's thesis project. Real posts are collected from the site, turned into a training set, and used to fine-tune [YugoGPT](https://huggingface.co/gordicaleksa/YugoGPT), a model pretrained for Serbian, Croatian, and Bosnian. At generation time you pick an engagement score from -1 (strongly disliked) to 1 (strongly liked). The model writes a confession aimed at that reaction.

## What it does

Each source post has approve and disapprove votes. Those votes become one score:

```text
engagement = (approve - disapprove) / (approve + disapprove + 1)
```

Training examples look like this:

```text
[engagement_score:0.8421][engagement:loved] Ispovest:
{confession text}</stop>
```

The `</stop>` token teaches the model where the post ends. Labels are `loved`, `liked`, `neutral`, `disliked`, and `hated`, cut from the score.

A small FastAPI app lets you set the score and sampling options, calls the fine-tuned model on Modal, and stores the result in SQLite.

## Pipeline

1. Collect confessions and vote counts into SQLite and JSONL.
2. Keep posts with enough votes and a clear approve/disapprove split, and drop texts that are too short or too long.
3. Attach the engagement score and the `</stop>` ending.
4. Fine-tune YugoGPT with QLoRA on a Modal A100. A second epoch is optional.
5. Generate from a prompt that contains only the score and the label. Inference runs on Modal.
6. Compare the base model, the 1-epoch adapter, and the 2-epoch adapter with perplexity, MAUVE, and a check that the requested engagement matches the text.

An earlier pass also grouped posts by topic with KeyBERT and UMAP. The engagement model above is trained on the filtered vote data, not on those topic labels.

## Stack

- Python
- YugoGPT, Hugging Face Transformers, Datasets, and PEFT
- QLoRA with bitsandbytes and PyTorch
- Modal for GPU training and inference
- FastAPI, Uvicorn, and SQLAlchemy
- SQLite

## Layout

| Path | Role |
| --- | --- |
| `Scraping/` | Collection from ispovesti.com |
| `data/` | SQLite database and row models |
| `training/` | Upload the JSONL to Modal and run QLoRA |
| `Test/` | Generation, benchmarks, and MAUVE |
| `web/` | Local UI: `uvicorn web.main:app --reload` |
| `docs/` | Thesis notes on training and evaluation |

The web app expects a deployed Modal generator (`Test/generate_confessions_by_engagement.py`) and a Modal token on the machine that serves the API.
