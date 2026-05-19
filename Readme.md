# guppyemail - Email Summarizer Prototype

A tiny (~9M parameter) language model trained **from scratch** to summarize emails. No PhD required. No massive GPU. One Colab notebook, 5 minutes, and you have your own LLM.

It won't write like a corporate assistant. But it'll tell you what the email says in a fun, conversational way.

```
Email: "Dear team, the Q3 budget review meeting has been moved from Feb 10 to Feb 5..."

guppyemail: "hey, meeting moved to feb 5. bring your reports. be there by friday."
```

## Features

- 📧 **Gmail Integration** - Connect your Gmail to fetch real emails
- **guppyemail Tiny LLM** - 9M params, trained from scratch, ~34MB checkpoint
- ⚡ **5-Minute Training** - Runs on free Colab GPU (T4)
- 📊 **Priority Classification** - Urgent / Important / Normal / Low
- ✅ **Action Item Extraction** - Finds tasks and deadlines
- 🌐 **Gradio Web UI** - Clean, interactive demo

## Quick Start

### 1. Setup

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate        # WSL/Linux
.venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Training Data

```bash
# Download Enron emails and generate summaries
export OPENAI_API_KEY="your-key"  # For summary generation (~$0.20 total)
python scripts/download_enron.py
python src/preprocess.py
python scripts/generate_summaries.py
```

### 3. Train the Model

Open `notebooks/train_email_guppylm.ipynb` in **Google Colab** (free GPU):
1. Upload training data from `data/training/`
2. Run all cells (~5 minutes)
3. Download `best_model.pt`, `config.json`, `tokenizer.json`

### 4. Run the App

```bash
# Place trained model files in checkpoints/
python app.py
```

Open `http://localhost:7860` in your browser.

## Project Structure

```
email-summarizer/
├── README.md
├── PROJECT-PLAN.md               # Full project plan
├── requirements.txt
├── config.py                     # Model + training config
├── model.py                      # guppyemail vanilla transformer
├── dataset.py                    # Data loading and batching
├── train.py                      # Training loop
├── inference.py                  # Chat/inference engine
├── notebooks/
│   └── train_email_guppylm.ipynb # Colab training notebook
├── src/
│   ├── gmail_client.py           # Gmail API wrapper
│   ├── preprocess.py             # Email cleaning pipeline
│   ├── classifier.py             # Priority classification
│   ├── action_extractor.py       # Action item extraction
│   └── pipeline.py               # End-to-end pipeline
├── scripts/
│   ├── download_enron.py         # Download Enron dataset
│   ├── generate_summaries.py     # LLM distillation for training data
│   └── evaluate.py               # ROUGE + perplexity evaluation
├── data/                         # Datasets (gitignored)
├── checkpoints/                  # Trained model weights
└── report/                       # Final report
```

## Architecture

| Component | Value |
|---|---|
| **Parameters** | 8.7M |
| **Architecture** | Vanilla transformer |
| **Layers** | 6 |
| **Hidden dim** | 384 |
| **Attention heads** | 6 |
| **FFN** | 768 (ReLU) |
| **Vocabulary** | 4,096 BPE tokens |
| **Max sequence** | 512 tokens |
| **Model size** | ~10MB |
| **Training time** | Depends |

No GQA, no RoPE, no SwiGLU. As simple as it gets.

## guppyemail

guppyemail is derived from the GuppyLM-style small-transformer architecture, but it is trained on a different email summarization dataset with a different scope.

- **A tiny LLM** - from tokenization to transformer to training loop
- **Training takes 5 minutes** - not 2-4 hours
- **It's fun** - quirky, conversational summaries make a memorable demo

The trade-off: summaries won't be polished corporate prose. They'll be short, casual, and a little weird. That's a feature.

## Evaluation

| Metric | Target |
|---|---|
| **Training loss** | < 3.0 |
| **ROUGE-L** | ≥ 0.20 |
| **Perplexity** | < 50 |
| **Inference time** | < 2 seconds (CPU) |
| **Model size** | < 20MB |

Run `python scripts/evaluate.py`.

## References

- [GuppyLM GitHub](https://github.com/arman-bd/guppylm)
- [GuppyLM HuggingFace](https://huggingface.co/arman-bd/guppylm-9M)
- [Article: "Build Your Own LLM in 5 Minutes"](https://arman-bd.medium.com/build-your-own-llm-in-5-minutes-i-made-mine-talk-like-a-fish-e20c338a3d14)
- [Datasets](https://huggingface.co/datasets/jacquelinehe/enron-emails)
- [Proccessed Data](https://huggingface.co/datasets/Siacelp/eron-mails-summarized)
