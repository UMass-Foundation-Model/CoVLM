# CoVLM: Composing Visual Entities and Relationships in Large Language Models Via Communicative Decoding

## News and ToDo List

* [ ] Release training scripts
* [ ] Release pre-training dataset
* [ ] Release demo
* [X] 2023-11-1: Release 1.4B/2.8B checkpoint
* [X] 2023-11-1: Release initial code

## Installation

```bash
conda create -n covlm python=3.9
conda activate covlm
# CUDA 10.2
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=10.2 -c pytorch
# CUDA 11.3
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
# CUDA 11.6
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
pip install -e transformers/
pip install -e YOLOX/
pip install -r requirements.txt
pip install -e .
python -m spacy download en_core_web_md
```

## Checkpoint

| Model      | Checkpoint                                                                        |
| ---------- | --------------------------------------------------------------------------------- |
| CoVLM-1.4B | [huggingface](https://huggingface.co/senfu/covlm-1.4b/blob/main/checkpoint_18000.pt) |
| CoVLM-2.8B | [huggingface](https://huggingface.co/senfu/covlm-2.8b/blob/main/checkpoint_15000.pt) |

## Evaluation

### RefCOCO/RefCOCOg/RefCOCOplus

```bash
bash eval_refcocog.sh CHECKPOINT
```
