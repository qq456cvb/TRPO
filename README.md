# TRPO

<!-- README refined by Cursor -->

TRPO in Tensorflow

## Overview

This repository contains Python code from an older research, course, or prototype project. The README has been refreshed to make the repository easier to scan while preserving the original notes below.

## Repository Contents

- Top-level source files and project assets.

## Setup

- This legacy repo does not pin a full environment. Start from the language/toolchain implied by the source files, then install missing packages as reported by the runtime.

## Usage

- inspect the top-level Python entry points: `atari_wrapper.py`, `common.py`, `simulator.py`, `train-atari.py`, `trpo.py`

## Data and Artifacts

No new large artifact is stored in this repository. If a dataset or checkpoint is required, follow the links and notes in the original section below.

## Status

This is a `Batch B` cleanup pass for a legacy repository. Commands may require dependency/version adjustments on a modern machine.

## License

See `LICENSE` for license details.

## Original Notes

# TRPO in TensorFlow
A Tensorflow implementation of "Trust Region Proximal Optimization" method. 

See the paper http://arxiv.org/abs/1502.05477

Currently working with discrete actions, continous(gaussian) variables support is straight forward.

# Features
Purely build on Tensorflow graphs and encapsulated as a seperate optimizer

You only need to pass the policy function and the cost function to the optimizer and create the cache variables.
