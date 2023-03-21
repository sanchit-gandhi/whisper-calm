# Whisper CALM

This is a repository for benchmarking the [Whisper Model](https://arxiv.org/abs/2212.04356) with early-exit decoding 
using the [Confident Adaptive Language Modeling (CALM)](https://arxiv.org/abs/2207.07061) framework by Schuster et al.

The code is split into two parts:
* [modeling_whisper_calm.py](modeling_whisper_calm.py): augments the Hugging Face Transformers Whisper implementation with early-exit decoding strategy computed via an entropy-based measure.
* [calibrate_whisper_calm.py](calibrate_whisper_calm.py): script for calibrating the Whisper model for textual consistency using the CALM framework.


## Background

Early exit is a paradigm for dynamically controlling the number of decoder layers used at inference time. It is based on the reasoning that the same amount of computation may not be required for every input to achieve adequate performance (i.e. depending on whether the input is easy or hard).

Instead of making a prediction based on the hidden-representation of the **final** decoder layer $\boldsymbol{d}\_{t}^{L}$, early exiting makes a prediction based on the hidden-representation for an **intermediate** layer $\boldsymbol{d}\_{t}^{i}$ for $i < L$. For each decoder layer $i$, we compute a confidence score $c_t^i$ for the $t$-th token. We also define an early-exit threshold $\lambda_t^i$. If our confidence score exceeds this threshold ($c_t^i > \lambda_t^i$), we exit early and greedily predict the most probable token:

$$ \hat{y}\_{t} = \text{argmax} \_{y_t} P(y_t | \boldsymbol{d}_t^i)$$

Otherwise, we continue to the next layer and repeat.

There are three questions that have to be answered with early exit:
1. What confidence measure to use?
2. How to set the exit threshold to achieve consistent performance with the full model?
3. How to handle missing hidden-representations due to early exit in previously predicted tokens?

The main paper followed for addressing these questions was Confident Adaptive Language Modeling (CALM), which builds on [Depth-Adaptive Transformer](https://arxiv.org/abs/1910.10073) by Elbayad et al. The results most relevant to Whisper are those from the textually consistent WMT task. See results boxed in red on image below. Textually consistent -> word ordering matters. WMT -> similar target length to speech (~30 tokens).

### What confidence measure to use?
CALM proposes three confidence measures:
1. Softmax diff: map the decoder hidden-state to the logit space ($W\_{emb} \boldsymbol{d}_t^i$), run a softmax to get probabilities $ \text{softmax} (\boldsymbol{W}\_{emb} \boldsymbol{d}_t^i)$ and take the difference between the top-2 most probable predictions. If large, the model is confident of its predictions and we can terminate. Requires us to run additional projections and top-k indexing (this was optimised for JAX on TPU in the original codebase)
2. Cosine sim: compute the cosine similarity between the representation for layer $i$ and layer $(i-1)$: $\cos (\boldsymbol{d}_t^i, \boldsymbol{d}_t^{i-1})$. If large, the decoder hidden-states have saturated and we can terminate early
3. Learned classifier: train a linear classifier to assign a confidence score: $c_t^i = \mathcal{M}(\boldsymbol{d}_t^i)$

-> I focussed on 1 & 2, since they’re parameter free and attain similar performance to the learned classifier approach. To bypass the top-k indexing in 1, I also used an entropy based measure.

### How to set the exit threshold?
CALM presents an algorithm for selecting an exit threshold that guarantees that the model will perform to within a certain tolerance of the full model with specified probability. I implemented this algorithm and initialised it with their suggested settings, but found that it was not super effective for Whisper, skipping close to zero layers. Instead, I swapped to performing a sweep over exit thresholds \lambda and recording the avg decoder layers and WER for each one. This is less rigorous, but gives us a good overall idea of the WER penalty incurred by skipping layers.

### How to handle missing hidden-representations?
CALM copies the decoder hidden-state for the exited decoder layer to all subsequent ones. This means that we still have to run the entire decoder forward even if we’ve skipped layers, but we can run the layers in parallel as soon as we know we have exited (since the input is the same for each one).
I wanted to gauge how many decoder layers we can skip with early exit, what the affect is on the WER performance, and thus how viable it is for Whisper. Therefore, I chose not to parallelise this computation in this first step, as this is only worthwhile if we can guarantee that the model retains its performance.

## Results

I benchmarked on 100 samples of the LibriSpeech-clean dataset -> this is the easiest ASR test set, and thus gives us an upper-bound for the number of decoder layers we can skip (since the model should be most confident). I used the Whisper medium model, which has a total of 24 decoder layers.

Top-2 diff:

| Threshold | Avg Decoder Layers | Runtime | E[Distance] | WER  | p_j |
|-----------|--------------------|---------|-------------|------|-----|
| 1.0000    | 24.0               | 44      | 0           | 2.30 | 0.0 |
| 0.9875    | 21.2               | 71.9    | 0.01        | 2.83 | 0.2 |
| 0.9750    | 21.0               | 72.2    | 0.01        | 3.42 | 0.2 |
| 0.9625    | 20.8               | 71.8    | 0.02        | 3.54 | 0.3 |
| 0.9500    | 20.7               | 71.8    | 0.02        | 3.60 | 0.3 |
| 0.9375    | 20.6               | 71.8    | 0.02        | 3.66 | 0.3 |
| 0.9250    | 20.5               | 71.7    | 0.03        | 4.30 | 0.4 |

Cosine Similarity:

| Threshold | Avg Decoder Layers | Runtime | E[Distance] | WER    | p_j |
|-----------|--------------------|---------|-------------|--------|-----|
| 1.0000    | 24.0               | 52.2    | 0           | 2.3    | 0.0 |
| 0.9875    | 23.2               | 54      | 0           | 2.71   | 0.2 |
| 0.9750    | 23.0               | 53.2    | 0.05        | 5.01   | 0.6 |
| 0.9625    | 22.8               | 52.5    | 0.04        | 5.48   | 0.5 |
| 0.9500    | 19.5               | 110     | 2.83        | 246.34 | 1.0 |

Entropy:

| Threshold | Avg Decoder Layers | Runtime | E[Distance] | WER  | p_j |
|-----------|--------------------|---------|-------------|------|-----|
| 1.0000    | 24.0               | 43.1    | 0           | 2.3  | 0.0 |
| 0.9875    | 21.0               | 69.8    | 0.01        | 2.83 | 0.2 |
| 0.9750    | 20.8               | 69.8    | 0.02        | 3.54 | 0.3 |
| 0.9625    | 20.5               | 69.3    | 0.03        | 4.07 | 0.4 |
| 0.9500    | 20.4               | 69.6    | 0.04        | 4.48 | 0.5 |

## Analysis

The model skips the last 4 layers almost immediately. However, the WER penalty is quite high even for just a 4-layer reduction. As we reduce the threshold, the number of layers skipped doesn’t reduce significantly, but the WER penalty continues to inflate.
Based on the best 4-layer reduction, we’d expect to get an inference speed up of about 1.2x if we optimised the code, but an absolute WER increase from 2.3% -> 3.6%. This is quite a heavy penalty for just a 1.2x speed-up.
It suggests to me that there is high-utilisation of the first 20 decoder layers in the pre-trained Whisper model, and that the final 4 layers are necessary for ensuring high transcription accuracy.
