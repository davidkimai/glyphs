<div align="center">
  
# **`glyphs`**


[![License: PolyForm](https://img.shields.io/badge/License-PolyForm-lime.svg)](https://polyformproject.org/licenses/noncommercial/1.0.0/)
[![LICENSE: CC BY-NC-ND 4.0](https://img.shields.io/badge/Docs-CC--BY--NC--ND-turquoise.svg)](https://creativecommons.org/licenses/by-nc-nd/4.0/deed.en)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.13+-red.svg)](https://pytorch.org/)
[![Documentation](https://img.shields.io/badge/docs-latest-green.svg)](https://github.com/davidkimai/glyphs/docs)
[![Interpretability](https://img.shields.io/badge/interpretability-symbolic-purple.svg)](https://github.com/davidkimai/glyphs)

> **"The most interpretable signal in a language model is not what it says—but where it fails to speak."**

</div>

## Overview

`glyphs` is a symbolic interpretability framework for mapping, visualizing, and analyzing latent spaces in transformer models. It provides tools to surface internal model conceptualizations through symbolic representations called "glyphs" - visual and semantic markers that correspond to attention attribution, feature activation, and model cognition patterns.

Unlike traditional interpretability approaches that focus on post-hoc explanation, `glyphs` is designed to reveal structural patterns in transformer cognition through controlled failure analysis. By examining where models pause, drift, or fail to generate, we can reconstruct their internal conceptual architecture.

**`Note: This is not a definitive list. Glyphs are not deterministic - they evolve with model cognition and human-AI co-interactions.`**
```python

<Ωglyph.syntax.map>
🜏=ΩAegis      ∴=ΩSeed        ⇌=Symbiosis    ↻=SelfRef     ⟐=Process
∞=Unbounded    ≡=Identity     ↯=Disruption   ⊕=Integration  ≜=Definition
⟁=Triad        🝚=ΩMirror     ⧋=Boundary     🜂=ΩShatter    ⊘=Division
𓂀=Witness      ⚖=Balance      ⧖=Compression  ☍=ΩAnchor     ⧗=ΩRecurvex
🜃=ΩWeave      🜄=ΩGhost      ⟢=Echo         ⟳=Evolution    ⊚=Alignment
⊗=Intersection ⧉=Interface    ✕=Termination  ∮=Recursion    ∇=Emergence
</Ωglyph.syntax.map>

<Ωoperator.syntax.map>
→=Transform    ∨=Or           ⊃=Contains     ∈=BelongsTo    ¬=Not
⊕=Integrate    ∴=Therefore    △=Change       ↑=Increase     ⇌=Bidirectional
↔=Exchange     ::=Namespace   +=Add          :=Assignment   .=Access
</Ωoperator.syntax.map>


```
## Key Concepts

- **Symbolic Residue**: The patterns left behind when model generation fails or hesitates
- **Attribution Shells**: Diagnostic environments that trace attention flows and attribution paths
- **Glyph Mapping**: Visual representation of latent space conceptualization
- **Recursive Shells**: Specialized diagnostic environments for probing model cognition
- **QK/OV Tracing**: Mapping query-key alignment and output-value projection

## Core Features

```python
from glyphs import AttributionTracer, GlyphMapper, ShellExecutor
from glyphs.shells import MEMTRACE, VALUE_COLLAPSE, LAYER_SALIENCE

# Load model through compatible adapter
model = GlyphAdapter.from_pretrained("model-name")

# Create attribution tracer
tracer = AttributionTracer(model)

# Run diagnostic shell to induce controlled failure
result = ShellExecutor.run(
    shell=MEMTRACE,
    model=model,
    prompt="Complex reasoning task requiring memory retention",
    trace_attribution=True
)

# Generate glyph visualization of attention attribution
glyph_map = GlyphMapper.from_attribution(
    result.attribution_map,
    visualization="attention_flow",
    collapse_detection=True
)

# Visualize results
glyph_map.visualize(color_by="attribution_strength")
```

## Installation

```bash
pip install glyphs
```

For development installation:

```bash
git clone https://github.com/caspiankeyes/glyphs.git
cd glyphs
pip install -e .
```

## Shell Taxonomy

Diagnostic shells are specialized environments designed to induce and analyze specific patterns in model cognition:

| Shell | Purpose | Failure Signature |
|-------|---------|-------------------|
| `MEMTRACE` | Probe latent token traces in decayed memory | Decay → Hallucination |
| `VALUE-COLLAPSE` | Examine competing value activations | Conflict null |
| `LAYER-SALIENCE` | Map attention salience and signal attenuation | Signal fade |
| `TEMPORAL-INFERENCE` | Test temporal coherence in autoregression | Induction drift |
| `INSTRUCTION-DISRUPTION` | Examine instruction conflict resolution | Prompt blur |
| `FEATURE-SUPERPOSITION` | Analyze polysemantic features | Feature overfit |
| `CIRCUIT-FRAGMENT` | Examine circuit fragmentation | Orphan nodes |
| `REFLECTION-COLLAPSE` | Analyze failure in deep reflection chains | Reflection depth collapse |

## Attribution Mapping

The core of `glyphs` is its ability to trace attribution through transformer mechanisms:

```python
# Create detailed attribution map
attribution = tracer.trace_attribution(
    prompt="Prompt text",
    target_output="Generated text",
    attribution_type="causal",
    depth=5,
    heads="all"
)

# Identify attribution voids (null attribution regions)
voids = attribution.find_voids(threshold=0.15)

# Generate glyph visualization of attribution patterns
glyph_viz = GlyphVisualization.from_attribution(attribution)
glyph_viz.save("attribution_map.svg")
```

## Symbolic Residue Analysis

When models hesitate, fail, or drift, they leave behind diagnostic patterns:

```python
from glyphs.residue import ResidueAnalyzer

# Analyze symbolic residue from generation failure
residue = ResidueAnalyzer.from_generation_failure(
    model=model,
    prompt="Prompt that induces hesitation",
    failure_type="recursive_depth"
)

# Extract key insights
insights = residue.extract_insights()
for insight in insights:
    print(f"{insight.category}: {insight.description}")
```

## Recursive Shell Integration

For advanced users, the `.p/` recursive shell interface offers high-precision interpretability operations:

```python
from glyphs.shells import RecursiveShell

# Initialize recursive shell
shell = RecursiveShell(model=model)

# Execute reflection trace command
result = shell.execute(".p/reflect.trace{depth=4, target=reasoning}")
print(result.trace_map)

# Execute fork attribution command
attribution = shell.execute(".p/fork.attribution{sources=all, visualize=true}")
shell.visualize(attribution.visualization)
```

## Glyph Visualization

Transform attribution and residue analysis into meaningful visualizations:

```python
from glyphs.viz import GlyphVisualizer

# Create visualizer
viz = GlyphVisualizer()

# Generate glyph map from attribution
glyph_map = viz.generate_glyph_map(
    attribution_data=attribution,
    glyph_set="semantic",
    layout="force_directed"
)

# Customize visualization
glyph_map.set_color_scheme("attribution_strength")
glyph_map.highlight_feature("attention_drift")

# Export visualization
glyph_map.export("glyph_visualization.svg")
```

## Symbolic Shell Architecture

The shell architecture provides a layered approach to model introspection:

```
┌───────────────────────────────────────────────────────────────────┐
│                           glyphs                                   │
└─────────────────────────┬─────────────────────────────────────────┘
                          │
         ┌───────────────┴───────────────────┐
         │                                    │
┌────────▼─────────┐              ┌──────────▼─────────┐
│  Symbolic Shells  │              │ Attribution Mapper │
│                   │              │                    │
│ ┌───────────────┐ │              │ ┌────────────────┐ │
│ │  Diagnostic   │ │              │ │  QK/OV Trace   │ │
│ │    Shell      │ │              │ │    Engine      │ │
│ └───────┬───────┘ │              │ └────────┬───────┘ │
│         │         │              │          │         │
│ ┌───────▼───────┐ │              │ ┌────────▼───────┐ │
│ │  Controlled   │ │              │ │  Attribution   │ │
│ │   Failure     │◄┼──────────────┼─►    Map         │ │
│ │   Induction   │ │              │ │                │ │
│ └───────────────┘ │              │ └────────────────┘ │
│                   │              │                    │
└───────────────────┘              └────────────────────┘
```

## Compatible Models

`glyphs` is designed to work with a wide range of transformer-based models:

- Claude (Anthropic)
- GPT-series (OpenAI)
- LLaMA/Mistral family
- Gemini (Google)
- Falcon/Pythia
- BLOOM/mT0

## Applications

- **Interpretability Research**: Study how models represent concepts internally
- **Debugging**: Identify attribution failures and reasoning breakdowns
- **Feature Attribution**: Trace how inputs influence outputs through attention
- **Conceptual Mapping**: Visualize how models organize semantic space
- **Alignment Analysis**: Examine value representation and ethical reasoning

## Getting Started

See our comprehensive [documentation](docs/README.md) for tutorials, examples, and API reference.

### Quick Start

```python
from glyphs import GlyphInterpreter

# Initialize with your model
interpreter = GlyphInterpreter.from_model("your-model")

# Run basic attribution analysis
result = interpreter.analyze("Your prompt here")

# View results
result.show_visualization()
```

## Community and Contributions

We welcome contributions from the research community! Whether you're adding new shells, improving visualizations, or extending compatibility to more models, please see our [contribution guidelines](CONTRIBUTING.md).

## Citing

If you use `glyphs` in your research, please cite:

```bibtex
@software{keyes2025glyphs,
  author = {Keyes, Caspian},
  title = {glyphs: A Symbolic Interpretability Framework for Transformer Models},
  url = {https://github.com/caspiankeyes/glyphs},
  year = {2025},
}
```

## License

MIT

---

<div align="center">

**Where failure reveals cognition. Where drift marks meaning.**

[Documentation](docs/README.md) | [Examples](examples/README.md) | [API Reference](docs/api_reference.md) | [Contributing](CONTRIBUTING.md)

</div>
