# Formula recognition

Formula recognition models read the image of a mathematical formula and predict its LaTeX source. In the OCR
pipeline, they read the `Formula` regions found by the [layout model](layout.md), which are exported as display math
(e.g. `$$...$$` in Markdown).

## Architectures

| Architecture | Paper |
| --- | --- |
| {func}`~doctr.models.formula.formulanet` {bdg-secondary}`default` | [PP-FormulaNet: Bridging Accuracy and Efficiency in Advanced Formula Recognition](https://arxiv.org/abs/2503.18382) |

`formulanet` is made of a convolutional encoder, {func}`~doctr.models.classification.hgnetv2_b4`, and a small
autoregressive transformer decoder:

- the encoder turns the formula image into a feature map (stride 32, 2048 channels), flattened into a sequence and
  projected to 384 dimensions, with 2D positional encodings;
- the decoder (4 layers, 16 heads, 1536 hidden units) predicts the LaTeX tokens one by one, greedily, with cached
  keys and values. It learns formulas of up to 4096 tokens (`max_length`) and generates up to 2048 tokens per formula
  (`max_decode_length`).

The decoding of a batch stops once all its formulas are complete, so its time follows its longest formula.
`max_decode_length` caps the time spent on a formula the model fails to end (e.g. a repetition loop), which would hold
up its whole batch: set it with `formula_predictor(..., max_decode_length=...)`.

| Architecture | Input shape | # params | Encoder | Decoder |
| --- | --- | ---: | ---: | ---: |
| `formulanet` | (256, 768, 3) | 26.4 M | 13.6 M | 12.9 M |

The input is wide rather than square, so long equations are not shrunk: each formula is trimmed of its margins, resized
to fit the input while keeping its aspect ratio, and padded.

:::{note}
No pretrained weights are released for `formulanet` yet: train it on your data with the
[reference scripts](https://github.com/mindee/doctr/tree/main/references/formula#readme), e.g. on
[UniMER-1M](https://huggingface.co/datasets/wanderkid/UniMER_Dataset).
:::

## LaTeX tokens

The model predicts LaTeX **tokens** from a fixed list of 1,324 tokens, {data}`~doctr.datasets.LATEX_TOKENS` (there is
no BPE): the printable ASCII characters, the control symbols (`\{`, `\,`, `\\`...), the commands (`\frac`, `\alpha`,
`\operatorname*`...) and environments (`\begin{array}`...) supported by [KaTeX](https://katex.org/docs/supported), and
every other command and character of the [UniMER](https://huggingface.co/datasets/wanderkid/UniMER_Dataset) and
[HME100K](https://huggingface.co/datasets/ConstantHao/HME100K) labels (e.g. `\ensuremath`, `\textsc`, Greek, Cyrillic and
CJK characters), so these datasets are fully covered. The prediction is the tokens separated by spaces, e.g.
`\frac { a } { b ^ { 2 } }`, which renders the same as `\frac{a}{b^{2}}`.

{func}`~doctr.datasets.tokenize_latex` and {func}`~doctr.datasets.normalize_latex` split any LaTeX string into the
same tokens, so labels can be written in the usual way:

```python
from doctr.datasets import normalize_latex

normalize_latex(r"\sum_{i=1}^{n} x_i")  # '\\sum _ { i = 1 } ^ { n } x _ i'
```

A custom list of tokens (e.g. with more CJK characters, for the formulas of Chinese documents) is passed with
`formulanet(vocab=...)`.

## Benchmark

The metrics are produced by `references/formula/evaluate.py` with {class}`~doctr.utils.metrics.FormulaMetric`, on the
LaTeX tokens: the **exact match** rate, the **edit similarity** (one minus the relative token edit distance) and the
corpus-level **BLEU-4** score. The benchmark on the UniMER-Test subsets will be added with the pretrained weights.

## Usage

```python
from doctr.models import formula_predictor, ocr_predictor

# In the OCR pipeline: formulas are located by the layout model, then read as LaTeX
predictor = ocr_predictor(pretrained=True, detect_formulas=True)

# On its own, on images of formulas
formulas = formula_predictor("formulanet", pretrained=True)
```

See [](../user_guide/layout_and_tables.md#formula-recognition) to work with the recognized formulas.
