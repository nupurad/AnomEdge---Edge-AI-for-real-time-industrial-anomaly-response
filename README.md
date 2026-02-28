# On-Device Factory Anomaly Detection Agent

A hackathon-ready, on-device pipeline for factory safety anomalies:

1. Camera frame is captured
2. Vision model classifies anomaly (`oil_spill`, `conveyor_jam`, `smoke`, `normal`)
3. Agent retrieves relevant SOP from local markdown docs (RAG)
4. Local LLM (Gemma-compatible) generates an action plan
5. Offline TTS speaks instructions to workers

## Why on-device

- Lower latency for safety-critical response
- Works with poor/no internet in industrial zones
- Privacy and bandwidth savings (no video upload)

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Build Training Data Fast (Hackathon)

Capture short class-wise videos on phone/webcam and place them in:

```text
data/source_videos/
  normal/
  oil_spill/
  conveyor_jam/
  smoke/
```

Then run:

```bash
# 1) Extract frames into data/raw/<class>/
python scripts/extract_frames.py --input-root data/source_videos --output-root data/raw --every-n-frames 8

# 2) Optional: boost minority class count
python scripts/augment_class.py --class-dir data/raw/smoke --copies-per-image 2

# 3) Split into train/val folders expected by trainer
python scripts/split_dataset.py --raw-root data/raw --out-root data/processed --val-ratio 0.2
```

### 1) Train MobileNet baseline

Expected dataset layout:

```text
data/processed/train/
  normal/
  oil_spill/
  conveyor_jam/
  smoke/
data/processed/val/
  normal/
  oil_spill/
  conveyor_jam/
  smoke/
```

Train:

```bash
python scripts/train_vision.py \
  --data-root data/processed \
  --epochs 5 \
  --batch-size 16 \
  --out-model models/mobilenet_anomaly.pt
```

### 2) Run live loop (camera + local agent + voice)

```bash
python scripts/run_agent.py \
  --vision-model models/mobilenet_anomaly.pt \
  --knowledge-dir knowledge \
  --llm-model-path models/gemma-2b-it-q4.gguf \
  --camera-index 0
```

## Architecture

- `src/vision/`: training + inference for MobileNet anomaly classifier
- `src/agent/`: markdown RAG and local LLM wrapper
- `src/audio/`: offline TTS output
- `src/pipeline/`: end-to-end camera loop orchestration
- `knowledge/`: local SOP markdown docs used for retrieval

## Hackathon demo tips (7 hours)

1. Start with transfer learning and a tiny balanced dataset.
2. Use synthetic augmentations for smoke/oil visual variety.
3. Keep LLM generation short (`max_tokens=120`) for low latency.
4. Cache retrieval chunks to avoid repeated parsing.
5. Data target: minimum 100-200 images/class (after extraction + augmentation) for a usable baseline.

## Notes

- LLM integration uses `llama-cpp-python` so any Gemma GGUF model that runs locally can be used.
- If no LLM model is present, fallback response is returned from retrieved SOP chunks.
