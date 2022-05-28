# AMOM

## Introduction

This repo contains the code of AMOM, which can be applied to various sequence to sequence tasks.

## Requirements and Installation
* [PyTorch](http://pytorch.org/) version == 1.10
* Python version >= 3.6
* Fairseq == 1.0.0
* sacrebleu==1.5.1
* sacremoses
* tensorboardX
* tensorboard

To install fairseq from source and develop locally:
```
cd fairseq-main
pip install --editable ./
```

## Examples

* [Nerual Translation Task](fairseq-main/examples/amom/translation/README.md).
* [Summarization Task](fairseq-main/examples/amom/summarization/README.md).
* [Code Generation Task](fairseq-main/examples/amom/code_generation/README.md).
