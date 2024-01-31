# cmmd-pytorch

(Unofficial) PyTorch implementation of CLIP Maximum Mean Discrepancy (CMMD) for evaluating image generation models, proposed in [Rethinking FID: Towards a Better Evaluation Metric for Image Generation](https://arxiv.org/abs/2401.09603). CMMD stands out to be a better metric than FID and tries to mitigate the longstanding issues of FID.

This implementation is a super simple PyTorch port of the [original codebase](https://github.com/google-research/google-research/tree/master/cmmd). I have only focused on the JAX and TensorFlow specific bits and replaced them PyTorch. Some differences:

* The original codebase relies on [`scenic`](https://github.com/google-research/scenic) for computing CLIP embeddings. This repository uses [`transformers`](https://github.com/huggingface/transformers).
* For the data loading, the original codebase uses TensorFlow, this one uses PyTorch `Dataset` and `DataLoader`.

## Setup

First, install PyTorch following instructions from the [official website](https://pytorch.org/).

Then install the depdencies:

```bash
pip install -r requirements.txt
```

## Running

```bash
python main.py /path/to/reference/images /path/to/eval/images --batch_size=32 --max_count=30000
```

A working example command:

```bash
python main.py reference_images generated_images --batch_size=1
```

> [!TIP]
> GPU execution is supported when a GPU is available.

## TODO

- [ ] Report a CMMD for SDXL on PartiPrompts
- [ ] Report a CMMD for SSD-1B on PartiPrompts
- [ ] Report a CMMD for PixArt-Alpha on PartiPrompts
- [ ] Report a CMMD for SDXL-Turbo on PartiPrompts


