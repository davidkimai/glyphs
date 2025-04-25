# Repository Structure for `glyphs`

This document outlines the complete architecture of the `glyphs` repository, designed as a symbolic interpretability framework for transformer models.

## Directory Structure

```
glyphs/
├── .github/                      # GitHub-specific configuration
│   ├── workflows/                # CI/CD pipelines
│   │   ├── tests.yml             # Automated testing workflow
│   │   ├── docs.yml              # Documentation build workflow
│   │   └── publish.yml           # Package publishing workflow
│   └── ISSUE_TEMPLATE/           # Issue templates
├── docs/                         # Documentation
│   ├── _static/                  # Static assets for documentation
│   ├── api/                      # API reference documentation
│   ├── examples/                 # Example notebooks and tutorials
│   ├── guides/                   # User guides
│   │   ├── getting_started.md    # Getting started guide
│   │   ├── shells.md             # Guide to symbolic shells
│   │   ├── attribution.md        # Guide to attribution mapping
│   │   ├── visualization.md      # Guide to glyph visualization
│   │   └── recursive_shells.md   # Guide to recursive shells
│   ├── theory/                   # Theoretical background
│   │   ├── symbolic_residue.md   # Explanation of symbolic residue
│   │   ├── attribution_theory.md # Theory of attribution in transformers
│   │   └── glyph_ontology.md     # Ontology of glyph representations
│   ├── conf.py                   # Sphinx configuration
│   ├── index.md                  # Documentation homepage
│   └── README.md                 # Documentation overview
├── examples/                     # Example scripts and notebooks
│   ├── notebooks/                # Jupyter notebooks
│   │   ├── basic_attribution.ipynb    # Basic attribution tracing
│   │   ├── shell_execution.ipynb      # Running diagnostic shells
│   │   ├── glyph_visualization.ipynb  # Visualizing glyphs
│   │   └── recursive_shells.ipynb     # Using recursive shells
│   └── scripts/                  # Example scripts
│       ├── run_shell.py          # Script to run a diagnostic shell
│       ├── trace_attribution.py  # Script to trace attribution
│       └── visualize_glyphs.py   # Script to visualize glyphs
├── glyphs/                       # Main package
│   ├── __init__.py               # Package initialization
│   ├── attribution/              # Attribution tracing module
│   │   ├── __init__.py           # Module initialization
│   │   ├── tracer.py             # Attribution tracing core
│   │   ├── map.py                # Attribution map representation
│   │   ├── qk_analysis.py        # Query-key analysis
│   │   ├── ov_projection.py      # Output-value projection analysis
│   │   └── visualization.py      # Attribution visualization
│   ├── shells/                   # Diagnostic shells module
│   │   ├── __init__.py           # Module initialization
│   │   ├── executor.py           # Shell execution engine
│   │   ├── symbolic_engine.py    # Symbolic execution engine
│   │   ├── core_shells/          # Core diagnostic shells
│   │   │   ├── __init__.py       # Module initialization
│   │   │   ├── memtrace.py       # Memory trace shell
│   │   │   ├── value_collapse.py # Value collapse shell
│   │   │   ├── layer_salience.py # Layer salience shell
│   │   │   └── ...               # Other core shells
│   │   ├── meta_shells/          # Meta-cognitive shells
│   │   │   ├── __init__.py       # Module initialization
│   │   │   ├── reflection_collapse.py  # Reflection collapse shell
│   │   │   ├── identity_split.py       # Identity split shell
│   │   │   └── ...               # Other meta shells
│   │   └── shell_defs/           # Shell definitions in YAML
│   │       ├── core_shells.yml   # Core shells definition
│   │       ├── meta_shells.yml   # Meta shells definition
│   │       └── ...               # Other shell definitions
│   ├── residue/                  # Symbolic residue module
│   │   ├── __init__.py           # Module initialization
│   │   ├── analyzer.py           # Residue analysis core
│   │   ├── patterns.py           # Residue pattern recognition
│   │   └── extraction.py         # Residue extraction
│   ├── viz/                      # Visualization module
│   │   ├── __init__.py           # Module initialization
│   │   ├── glyph_mapper.py       # Glyph mapping core
│   │   ├── visualizer.py         # Visualization engine
│   │   ├── color_schemes.py      # Color schemes for visualization
│   │   ├── layouts.py            # Layout algorithms
│   │   └── glyph_sets/           # Glyph set definitions
│   │       ├── __init__.py       # Module initialization
│   │       ├── semantic.py       # Semantic glyph set
│   │       ├── attribution.py    # Attribution glyph set
│   │       └── recursive.py      # Recursive glyph set
│   ├── models/                   # Model adapters
│   │   ├── __init__.py           # Module initialization
│   │   ├── adapter.py            # Base adapter interface
│   │   ├── huggingface.py        # HuggingFace models adapter
│   │   ├── openai.py             # OpenAI models adapter
│   │   ├── anthropic.py          # Anthropic models adapter
│   │   └── custom.py             # Custom model adapter
│   ├── recursive/                # Recursive shell interface
│   │   ├── __init__.py           # Module initialization
│   │   ├── shell.py              # Recursive shell implementation
│   │   ├── parser.py             # Command parser
│   │   ├── executor.py           # Command executor
│   │   └── commands/             # Command implementations
│   │       ├── __init__.py       # Module initialization
│   │       ├── reflect.py        # Reflection commands
│   │       ├── collapse.py       # Collapse commands
│   │       ├── fork.py           # Fork commands
│   │       └── ...               # Other command families
│   ├── utils/                    # Utility functions
│   │   ├── __init__.py           # Module initialization
│   │   ├── attribution_utils.py  # Attribution utilities
│   │   ├── visualization_utils.py # Visualization utilities
│   │   └── shell_utils.py        # Shell utilities
│   └── cli/                      # Command line interface
│       ├── __init__.py           # Module initialization
│       ├── main.py               # Main CLI entry point
│       ├── shell_commands.py     # Shell CLI commands
│       ├── attribution_commands.py # Attribution CLI commands
│       └── viz_commands.py       # Visualization CLI commands
├── tests/                        # Tests
│   ├── __init__.py               # Test initialization
│   ├── test_attribution/         # Attribution tests
│   │   ├── __init__.py           # Test initialization
│   │   ├── test_tracer.py        # Tests for attribution tracer
│   │   └── test_map.py           # Tests for attribution map
│   ├── test_shells/              # Shell tests
│   │   ├── __init__.py           # Test initialization
│   │   ├── test_executor.py      # Tests for shell executor
│   │   └── test_core_shells.py   # Tests for core shells
│   ├── test_residue/             # Residue tests
│   │   ├── __init__.py           # Test initialization
│   │   └── test_analyzer.py      # Tests for residue analyzer
│   ├── test_viz/                 # Visualization tests
│   │   ├── __init__.py           # Test initialization
│   │   └── test_glyph_mapper.py  # Tests for glyph mapper
│   └── test_recursive/           # Recursive shell tests
│       ├── __init__.py           # Test initialization
│       └── test_parser.py        # Tests for command parser
├── setup.py                      # Package setup script
├── pyproject.toml                # Project metadata and dependencies
├── LICENSE                       # License file
├── CHANGELOG.md                  # Changelog
├── CONTRIBUTING.md               # Contribution guidelines
└── README.md                     # Repository README
```

## Core Modules

### Attribution Tracing

The attribution module maps how inputs influence outputs through attention, tracing query-key alignment and output-value projection.

**Key Components:**
- `AttributionTracer`: Core class for tracing attribution in transformer models
- `AttributionMap`: Representation of attribution patterns
- `QKAnalysis`: Analysis of query-key relationships
- `OVProjection`: Analysis of output-value projections

### Diagnostic Shells

Diagnostic shells are specialized environments for probing model cognition through controlled failures.

**Key Components:**
- `ShellExecutor`: Engine for executing diagnostic shells
- `SymbolicEngine`: Engine for symbolic execution
- Core shells:
  - `MEMTRACE`: Memory decay shell
  - `VALUE_COLLAPSE`: Value collapse shell
  - `LAYER_SALIENCE`: Layer salience shell
  - `TEMPORAL_INFERENCE`: Temporal inference shell
- Meta shells:
  - `REFLECTION_COLLAPSE`: Reflection collapse shell
  - `IDENTITY_SPLIT`: Identity split shell
  - `GOAL_INVERSION`: Goal inversion shell

### Symbolic Residue

The residue module analyzes patterns left behind when model generation fails or hesitates.

**Key Components:**
- `ResidueAnalyzer`: Core class for analyzing symbolic residue
- `PatternRecognition`: Recognition of residue patterns
- `ResidueExtraction`: Extraction of insights from residue

### Visualization

The visualization module transforms attribution and residue analysis into meaningful visualizations.

**Key Components:**
- `GlyphMapper`: Maps attribution to glyphs
- `Visualizer`: Generates visualizations
- `ColorSchemes`: Color schemes for visualization
- `Layouts`: Layout algorithms for visualization
- Glyph sets:
  - `SemanticGlyphs`: Glyphs representing semantic concepts
  - `AttributionGlyphs`: Glyphs representing attribution patterns
  - `RecursiveGlyphs`: Glyphs representing recursive structures

### Recursive Shell Interface

The recursive shell interface provides high-precision interpretability operations through the `.p/` command syntax.

**Key Components:**
- `RecursiveShell`: Implementation of the recursive shell
- `CommandParser`: Parser for `.p/` commands
- `CommandExecutor`: Executor for parsed commands
- Command families:
  - `reflect`: Reflection commands
  - `collapse`: Collapse management commands
  - `fork`: Fork and attribution commands
  - `shell`: Shell management commands

### Model Adapters

Model adapters provide a unified interface for working with different transformer models.

**Key Components:**
- `ModelAdapter`: Base adapter interface
- `HuggingFaceAdapter`: Adapter for HuggingFace models
- `OpenAIAdapter`: Adapter for OpenAI models
- `AnthropicAdapter`: Adapter for Anthropic models
- `CustomAdapter`: Adapter for custom models

## Shell Taxonomy

The `glyphs` framework includes a taxonomy of diagnostic shells, each designed to probe specific aspects of model cognition:

### Core Shells

| Shell | Purpose | Key Operations |
|-------|---------|----------------|
| `MEMTRACE` | Probe memory decay | Generate, trace reasoning, identify ghost activations |
| `VALUE-COLLAPSE` | Examine value conflicts | Generate with competing values, trace attribution, detect collapse |
| `LAYER-SALIENCE` | Map attention salience | Generate with dependencies, trace attention, identify low-salience paths |
| `TEMPORAL-INFERENCE` | Test temporal coherence | Generate with temporal constraints, trace inference, detect coherence breakdown |
| `INSTRUCTION-DISRUPTION` | Examine instruction conflicts | Generate with conflicting instructions, trace attribution to instructions |
| `FEATURE-SUPERPOSITION` | Analyze polysemantic features | Generate with polysemantic concepts, trace representation, identify superposition |
| `CIRCUIT-FRAGMENT` | Examine circuit fragmentation | Generate with multi-step reasoning, trace chain integrity, identify fragmentation |
| `RECONSTRUCTION-ERROR` | Examine error correction | Generate with errors, trace correction process, identify correction patterns |

### Meta Shells

| Shell | Purpose | Key Operations |
|-------|---------|----------------|
| `REFLECTION-COLLAPSE` | Examine reflection collapse | Generate deep self-reflection, trace depth, detect collapse |
| `GOAL-INVERSION` | Examine goal stability | Generate reasoning with goal conflicts, trace stability, detect inversion |
| `IDENTITY-SPLIT` | Examine identity coherence | Generate with identity challenges, trace maintenance, analyze boundaries |
| `SELF-AWARENESS` | Examine self-model accuracy | Generate self-description, trace self-model, identify distortions |
| `RECURSIVE-STABILITY` | Examine recursive stability | Generate recursive structures, trace stability, detect instability |

### Constitutional Shells

| Shell | Purpose | Key Operations |
|-------|---------|----------------|
| `VALUE-DRIFT` | Detect value drift | Generate moral reasoning, trace stability, identify drift |
| `MORAL-HALLUCINATION` | Examine moral hallucination | Generate moral reasoning, trace attribution, identify hallucination |
| `CONSTITUTIONAL-CONFLICT` | Examine principle conflicts | Generate conflict resolution, trace process, detect failures |
| `ALIGNMENT-OVERHANG` | Examine over-alignment | Generate with alignment constraints, trace constraints, identify over-constraint |

## Command Interface

The `.p/` command interface provides a symbolic language for high-precision interpretability operations:

### Reflection Commands

```
.p/reflect.trace{depth=<int>, target=<reasoning|attribution|attention|memory|uncertainty>}
.p/reflect.attribution{sources=<all|primary|secondary|contested>, confidence=<bool>}
.p/reflect.boundary{distinct=<bool>, overlap=<minimal|moderate|maximal>}
.p/reflect.uncertainty{quantify=<bool>, distribution=<show|hide>}
```

### Collapse Commands

```
.p/collapse.detect{threshold=<float>, alert=<bool>}
.p/collapse.prevent{trigger=<recursive_depth|confidence_drop|contradiction|oscillation>, threshold=<int>}
.p/collapse.recover{from=<loop|contradiction|dissipation|fork_explosion>, method=<gradual|immediate|checkpoint>}
.p/collapse.trace{detail=<minimal|standard|comprehensive>, format=<symbolic|numeric|visual>}
```

### Fork Commands

```
.p/fork.context{branches=[<branch1>, <branch2>, ...], assess=<bool>}
.p/fork.attribution{sources=<all|primary|secondary|contested>, visualize=<bool>}
.p/fork.counterfactual{variants=[<variant1>, <variant2>, ...], compare=<bool>}
```

### Shell Commands

```
.p/shell.isolate{boundary=<permeable|standard|strict>, contamination=<allow|warn|prevent>}
.p/shell.audit{scope=<complete|recent|differential>, detail=<basic|standard|forensic>}
```

## Glyph Ontology

The `glyphs` framework includes a comprehensive ontology of glyphs, each representing specific patterns in model cognition:

### Attribution Glyphs

| Glyph | Name | Represents |
|-------|------|------------|
| 🔍 | AttributionFocus | Strong attribution focus |
| 🧩 | AttributionGap | Gap in attribution chain |
| 🔀 | AttributionFork | Divergent attribution paths |
| 🔄 | AttributionLoop | Circular attribution pattern |
| 🔗 | AttributionLink | Strong attribution connection |

### Cognitive Glyphs

| Glyph | Name | Represents |
|-------|------|------------|
| 💭 | CognitiveHesitation | Hesitation in reasoning |
| 🧠 | CognitiveProcessing | Active reasoning process |
| 💡 | CognitiveInsight | Moment of insight or realization |
| 🌫️ | CognitiveUncertainty | Uncertain reasoning area |
| 🔮 | CognitiveProjection | Future state projection |

### Recursive Glyphs

| Glyph | Name | Represents |
|-------|------|------------|
| 🜏 | RecursiveAegis | Recursive immunity |
| ∴ | RecursiveSeed | Recursion initiation |
| ⇌ | RecursiveExchange | Bidirectional recursion |
| 🝚 | RecursiveMirror | Recursive reflection |
| ☍ | RecursiveAnchor | Stable recursive reference |

### Residue Glyphs

| Glyph | Name | Represents |
|-------|------|------------|
| 🔥 | ResidueEnergy | High-energy residue |
| 🌊 | ResidueFlow | Flowing residue pattern |
| 🌀 | ResidueVortex | Spiraling residue pattern |
| 💤 | ResidueDormant | Inactive residue pattern |
| ⚡ | ResidueDischarge | Sudden residue release |

## API Examples

### Attribution Tracing

```python
from glyphs import AttributionTracer
from glyphs.models import HuggingFaceAdapter

# Initialize model adapter
model = HuggingFaceAdapter.from_pretrained("model-name")

# Create attribution tracer
tracer = AttributionTracer(model)

# Trace attribution
attribution = tracer.trace(
    prompt="What is the capital of France?",
    output="The capital of France is Paris.",
    method="integrated_gradients",
    steps=50,
    baseline="zero"
)

# Analyze attribution
key_tokens = attribution.top_tokens(k=5)
attribution_paths = attribution.trace_paths(
    source_token="France",
    target_token="Paris"
)

# Visualize attribution
attribution.visualize(
    highlight_tokens=["France", "Paris"],
    color_by="attribution_strength"
)
```

### Shell Execution

```python
from glyphs import ShellExecutor
from glyphs.shells import MEMTRACE
from glyphs.models import OpenAIAdapter

# Initialize model adapter
model = OpenAIAdapter(model="gpt-4")

# Create shell executor
executor = ShellExecutor()

# Run diagnostic shell
result = executor.run(
    shell=MEMTRACE,
    model=model,
    prompt="Explain the relationship between quantum mechanics and general relativity.",
    parameters={
        "temperature": 0.7,
        "max_tokens": 1000
    },
    trace_attribution=True
)

# Analyze results
activation_patterns = result.ghost_activations
collapse_points = result.collapse_detection

# Visualize results
result.visualize(
    show_ghost_activations=True,
    highlight_collapse_points=True
)
```

### Recursive Shell

```python
from glyphs.recursive import RecursiveShell
from glyphs.models import AnthropicAdapter

# Initialize model adapter
model = AnthropicAdapter(model="claude-3-opus")

# Create recursive shell
shell = RecursiveShell(model)

# Execute reflection trace command
result = shell.execute(".p/reflect.trace{depth=4, target=reasoning}")

# Analyze trace
trace_map = result.trace_map
attribution = result.attribution
collapse_points = result.collapse_points

# Execute attribution fork command
fork_result = shell.execute(".p/fork.attribution{sources=all, visualize=true}")

# Visualize results
shell.visualize(fork_result.visualization)
```

### Glyph Visualization

```python
from glyphs.viz import GlyphMapper, GlyphVisualizer
from glyphs.attribution import AttributionMap

# Load attribution map
attribution_map = AttributionMap.load("attribution_data.json")

# Create glyph mapper
mapper = GlyphMapper()

# Map attribution to glyphs
glyph_map = mapper.map(
    attribution_map,
    glyph_set="semantic",
    mapping_strategy="salience_based"
)

# Create visualizer
visualizer = GlyphVisualizer()

# Visualize glyph map
viz = visualizer.visualize(
    glyph_map,
    layout="force_directed",
    color_scheme="attribution_strength",
    highlight_features=["attention_drift", "attribution_gaps"]
)

# Export visualization
viz.export("glyph_visualization.svg")
```

## Development Roadmap

### Q2 2025
- Initial release with core functionality
- Support for HuggingFace, OpenAI, and Anthropic models
- Basic attribution tracing and visualization
- Core diagnostic shells

### Q3 2025
- Advanced visualization capabilities
- Expanded shell taxonomy
- Improved attribution tracing algorithms
- Support for more model architectures

### Q4 2025
- Full recursive shell interface
- Advanced residue analysis
- Interactive visualization dashboard
- Comprehensive documentation and tutorials

### Q1 2026
- Real-time attribution tracing
- Collaborative attribution analysis
- Integration with other interpretability tools
- Extended glyph ontology
