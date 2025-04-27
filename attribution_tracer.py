"""
attribution_tracer.py

Core implementation of the Attribution Tracing module for the glyphs framework.
This module maps token-to-token attribution flows, tracks query-key alignment,
and visualizes attention patterns to reveal latent semantic structures.
"""

import logging
import time
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any, Set
from dataclasses import dataclass, field
import json
import hashlib
from pathlib import Path
import matplotlib.pyplot as plt
import networkx as nx
from enum import Enum

from ..models.adapter import ModelAdapter
from ..utils.visualization_utils import VisualizationEngine

# Configure attribution-aware logging
logger = logging.getLogger("glyphs.attribution_tracer")
logger.setLevel(logging.INFO)

class AttributionType(Enum):
    """Types of attribution that can be traced."""
    DIRECT = "direct"               # Direct attribution between tokens
    INDIRECT = "indirect"           # Indirect attribution through intermediate tokens
    RESIDUAL = "residual"           # Attribution through residual connections
    MULTIHEAD = "multihead"         # Attribution through multiple attention heads
    NULL = "null"                   # No clear attribution path
    COMPOSITE = "composite"         # Mixture of multiple attribution types
    RECURSIVE = "recursive"         # Self-referential attribution path
    EMERGENT = "emergent"           # Emerged from collective behavior, not individual tokens


@dataclass
class AttributionLink:
    """A link in an attribution chain between source and target tokens."""
    source_idx: int              # Index of source token
    target_idx: int              # Index of target token
    attribution_type: AttributionType  # Type of attribution
    strength: float              # Strength of attribution (0.0-1.0)
    attention_heads: List[int] = field(default_factory=list)  # Contributing attention heads
    layers: List[int] = field(default_factory=list)  # Contributing layers
    intermediate_tokens: List[int] = field(default_factory=list)  # Intermediate tokens in indirect attribution
    residue: Optional[Dict[str, Any]] = None  # Symbolic residue if attribution is weak/null


@dataclass
class AttributionMap:
    """Complete map of attribution across a sequence."""
    prompt_tokens: List[str]         # Tokenized prompt
    output_tokens: List[str]         # Tokenized output
    links: List[AttributionLink]     # Attribution links
    token_salience: Dict[int, float] = field(default_factory=dict)  # Salience of each token
    attribution_gaps: List[Tuple[int, int]] = field(default_factory=list)  # Gaps in attribution
    collapsed_regions: List[Tuple[int, int]] = field(default_factory=list)  # Regions with attribution collapse
    uncertainty: Dict[int, float] = field(default_factory=dict)  # Uncertainty in attribution
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional metadata


@dataclass
class ForkPath:
    """A fork in the attribution path, representing alternative attributions."""
    id: str                         # Fork path ID
    description: str                # Description of the fork
    links: List[AttributionLink]    # Attribution links in this fork
    confidence: float               # Confidence in this fork (0.0-1.0)
    conflict_points: List[int] = field(default_factory=list)  # Tokens with conflicting attribution
    residue: Optional[Dict[str, Any]] = None  # Symbolic residue in the fork


@dataclass
class AttentionHead:
    """Representation of an attention head's behavior."""
    layer: int                      # Layer containing this head
    head: int                       # Head index
    pattern_type: str               # Type of attention pattern
    focus_tokens: List[int]         # Tokens this head focuses on
    strength: float                 # Overall attention strength
    function: Optional[str] = None  # Inferred function of this head
    attribution_role: Optional[str] = None  # Role in attribution process


class AttributionTracer:
    """
    Core attribution tracing system for the glyphs framework.
    
    This class implements attribution tracing between tokens, mapping
    how information flows through transformer architectures from inputs
    to outputs. It provides insights into the causal relationships
    between tokens and the formation of semantic structures.
    """
    
    def __init__(
        self,
        model: ModelAdapter,
        config: Optional[Dict[str, Any]] = None,
        visualizer: Optional[VisualizationEngine] = None
    ):
        """
        Initialize the attribution tracer.
        
        Parameters:
        -----------
        model : ModelAdapter
            Model adapter for the target model
        config : Optional[Dict[str, Any]]
            Configuration parameters for the tracer
        visualizer : Optional[VisualizationEngine]
            Visualization engine for attribution visualization
        """
        self.model = model
        self.config = config or {}
        self.visualizer = visualizer
        
        # Configure tracer parameters
        self.trace_depth = self.config.get("trace_depth", 5)
        self.min_attribution_strength = self.config.get("min_attribution_strength", 0.1)
        self.include_indirect = self.config.get("include_indirect", True)
        self.trace_residual = self.config.get("trace_residual", True)
        self.collapse_threshold = self.config.get("collapse_threshold", 0.05)
        
        # Track traced attribution maps
        self.attribution_history = []
        
        # Initialize glyph mappings for attribution
        self._init_attribution_glyphs()
        
        logger.info(f"Attribution tracer initialized for model: {model.model_id}")
    
    def _init_attribution_glyphs(self):
        """Initialize glyph mappings for attribution visualization."""
        # Attribution strength glyphs
        self.strength_glyphs = {
            "very_strong": "🔍",  # Very strong attribution (0.8-1.0)
            "strong": "🔗",      # Strong attribution (0.6-0.8)
            "moderate": "🧩",    # Moderate attribution (0.4-0.6)
            "weak": "🌫️",        # Weak attribution (0.2-0.4)
            "very_weak": "💤",   # Very weak attribution (0.0-0.2)
        }
        
        # Attribution type glyphs
        self.type_glyphs = {
            AttributionType.DIRECT: "⮕",      # Direct attribution
            AttributionType.INDIRECT: "⤑",     # Indirect attribution
            AttributionType.RESIDUAL: "↝",    # Residual attribution
            AttributionType.MULTIHEAD: "⥇",   # Multihead attribution
            AttributionType.NULL: "⊘",        # Null attribution
            AttributionType.COMPOSITE: "⬥",   # Composite attribution
            AttributionType.RECURSIVE: "↻",   # Recursive attribution
            AttributionType.EMERGENT: "⇞",    # Emergent attribution
        }
        
        # Pattern glyphs
        self.pattern_glyphs = {
            "attribution_chain": "🔗",        # Clean attribution chain
            "attribution_fork": "🔀",         # Forking attribution
            "attribution_loop": "🔄",         # Loop in attribution
            "attribution_gap": "⊟",          # Gap in attribution
            "attribution_cluster": "☷",      # Cluster of attribution
            "attribution_decay": "🌊",        # Attribution decay
            "attribution_conflict": "⚡",     # Conflicting attribution
        }
        
        # Meta glyphs
        self.meta_glyphs = {
            "attribution_focus": "🎯",        # Attribution focal point
            "uncertainty": "❓",              # Uncertainty in attribution
            "recursive_reference": "🜏",      # Recursive attribution reference
            "collapse_point": "🝚",           # Attribution collapse point
        }
    
    def trace(
        self,
        prompt: str,
        output: Optional[str] = None,
        depth: Optional[int] = None,
        include_confidence: bool = True,
        visualize: bool = False
    ) -> AttributionMap:
        """
        Trace attribution between prompt and output.
        
        Parameters:
        -----------
        prompt : str
            Input prompt
        output : Optional[str]
            Output to trace attribution for. If None, will generate output.
        depth : Optional[int]
            Depth of attribution tracing. If None, uses default.
        include_confidence : bool
            Whether to include confidence scores
        visualize : bool
            Whether to generate visualization
            
        Returns:
        --------
        AttributionMap
            Map of attribution between tokens
        """
        trace_start = time.time()
        depth = depth or self.trace_depth
        
        logger.info(f"Tracing attribution with depth {depth}")
        
        # Generate output if not provided
        if output is None:
            output = self.model.generate(prompt=prompt, max_tokens=800)
        
        # Get token-level representations
        prompt_tokens = self._tokenize(prompt)
        output_tokens = self._tokenize(output)
        
        # Create attribution map
        attribution_map = AttributionMap(
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            links=[],
            metadata={
                "prompt": prompt,
                "output": output,
                "model_id": self.model.model_id,
                "trace_depth": depth,
                "timestamp": time.time()
            }
        )
        
        # If direct API access is available, get attribution directly
        if hasattr(self.model, "get_attribution"):
            try:
                logger.info("Getting attribution directly from model API")
                api_attribution = self.model.get_attribution(
                    prompt=prompt,
                    output=output,
                    include_confidence=include_confidence
                )
                attribution_map = self._process_api_attribution(
                    api_attribution,
                    prompt_tokens,
                    output_tokens
                )
                logger.info("Successfully processed API attribution")
            except Exception as e:
                logger.warning(f"Failed to get attribution from API: {e}")
                logger.info("Falling back to inference-based attribution")
                attribution_map = self._infer_attribution(
                    prompt=prompt,
                    output=output,
                    prompt_tokens=prompt_tokens,
                    output_tokens=output_tokens,
                    depth=depth
                )
        else:
            # Fall back to inference-based attribution
            logger.info("Using inference-based attribution")
            attribution_map = self._infer_attribution(
                prompt=prompt,
                output=output,
                prompt_tokens=prompt_tokens,
                output_tokens=output_tokens,
                depth=depth
            )
        
        # Analyze token salience
        attribution_map.token_salience = self._analyze_token_salience(
            attribution_map.links
        )
        
        # Find attribution gaps
        attribution_map.attribution_gaps = self._find_attribution_gaps(
            attribution_map.links,
            len(prompt_tokens),
            len(output_tokens)
        )
        
        # Detect collapsed regions
        attribution_map.collapsed_regions = self._detect_collapsed_regions(
            attribution_map.links,
            len(prompt_tokens),
            len(output_tokens)
        )
        
        # Calculate uncertainty if requested
        if include_confidence:
            attribution_map.uncertainty = self._calculate_attribution_uncertainty(
                attribution_map.links
            )
        
        # Generate visualization if requested
        if visualize and self.visualizer:
            visualization = self.visualizer.visualize_attribution(attribution_map)
            attribution_map.metadata["visualization"] = visualization
        
        # Record execution time
        trace_time = time.time() - trace_start
        attribution_map.metadata["trace_time"] = trace_time
        
        # Add to history
        self.attribution_history.append(attribution_map)
        
        logger.info(f"Attribution tracing completed in {trace_time:.2f}s")
        return attribution_map
    
    def trace_with_forks(
        self,
        prompt: str,
        output: Optional[str] = None,
        fork_factor: int = 3,
        include_confidence: bool = True,
        visualize: bool = False
    ) -> Dict[str, Any]:
        """
        Trace attribution with multiple fork paths.
        
        Parameters:
        -----------
        prompt : str
            Input prompt
        output : Optional[str]
            Output to trace attribution for. If None, will generate output.
        fork_factor : int
            Number of alternative attribution paths to generate
        include_confidence : bool
            Whether to include confidence scores
        visualize : bool
            Whether to generate visualization
            
        Returns:
        --------
        Dict[str, Any]
            Dictionary containing attribution map and fork paths
        """
        trace_start = time.time()
        
        logger.info(f"Tracing attribution with {fork_factor} fork paths")
        
        # Generate output if not provided
        if output is None:
            output = self.model.generate(prompt=prompt, max_tokens=800)
        
        # Get token-level representations
        prompt_tokens = self._tokenize(prompt)
        output_tokens = self._tokenize(output)
        
        # Get base attribution map
        base_attribution = self.trace(
            prompt=prompt,
            output=output,
            include_confidence=include_confidence,
            visualize=False
        )
        
        # Generate fork paths
        fork_paths = []
        
        # Identify conflict points for forking
        conflict_points = self._identify_conflict_points(base_attribution)
        
        # Generate alternative attribution paths
        for i in range(fork_factor):
            # Create fork with alternative attributions
            fork_path = self._generate_fork_path(
                base_attribution=base_attribution,
                conflict_points=conflict_points,
                fork_id=f"fork_{i+1}",
                fork_index=i
            )
            fork_paths.append(fork_path)
        
        # Create fork result
        fork_result = {
            "base_attribution": base_attribution,
            "fork_paths": fork_paths,
            "conflict_points": conflict_points,
            "metadata": {
                "prompt": prompt,
                "output": output,
                "model_id": self.model.model_id,
                "fork_factor": fork_factor,
                "timestamp": time.time()
            }
        }
        
        # Generate visualization if requested
        if visualize and self.visualizer:
            visualization = self.visualizer.visualize_attribution_forks(fork_result)
            fork_result["metadata"]["visualization"] = visualization
        
        # Record execution time
        trace_time = time.time() - trace_start
        fork_result["metadata"]["trace_time"] = trace_time
        
        logger.info(f"Attribution fork tracing completed in {trace_time:.2f}s")
        return fork_result
    
    def trace_attention_heads(
        self,
        prompt: str,
        output: Optional[str] = None,
        layer_range: Optional[Tuple[int, int]] = None,
        head_threshold: float = 0.1,
        visualize: bool = False
    ) -> Dict[str, Any]:
        """
        Trace attribution through specific attention heads.
        
        Parameters:
        -----------
        prompt : str
            Input prompt
        output : Optional[str]
            Output to trace attention for. If None, will generate output.
        layer_range : Optional[Tuple[int, int]]
            Range of layers to analyze. If None, analyzes all layers.
        head_threshold : float
            Minimum attention strength to include head
        visualize : bool
            Whether to generate visualization
            
        Returns:
        --------
        Dict[str, Any]
            Dictionary containing attention head analysis
        """
        trace_start = time.time()
        
        # Generate output if not provided
        if output is None:
            output = self.model.generate(prompt=prompt, max_tokens=800)
        
        # Get token-level representations
        prompt_tokens = self._tokenize(prompt)
        output_tokens = self._tokenize(output)
        
        # Get model information
        model_info = self.model.get_model_info()
        num_layers = model_info.get("num_layers", 12)
        num_heads = model_info.get("num_heads", 12)
        
        # Determine layer range
        if layer_range is None:
            layer_range = (0, num_layers - 1)
        else:
            layer_range = (
                max(0, layer_range[0]),
                min(num_layers - 1, layer_range[1])
            )
        
        # Trace attention head behavior
        attention_heads = []
        
        # If direct API access is available, get attention patterns directly
        if hasattr(self.model, "get_attention_patterns"):
            try:
                logger.info("Getting attention patterns directly from model API")
                attention_patterns = self.model.get_attention_patterns(
                    prompt=prompt,
                    output=output,
                    layer_range=layer_range
                )
                attention_heads = self._process_api_attention(
                    attention_patterns,
                    prompt_tokens,
                    output_tokens,
                    layer_range,
                    head_threshold
                )
                logger.info("Successfully processed API attention patterns")
            except Exception as e:
                logger.warning(f"Failed to get attention patterns from API: {e}")
                logger.info("Falling back to inference-based attention analysis")
                attention_heads = self._infer_attention_behavior(
                    prompt=prompt,
                    output=output,
                    prompt_tokens=prompt_tokens,
                    output_tokens=output_tokens,
                    layer_range=layer_range,
                    num_heads=num_heads,
                    head_threshold=head_threshold
                )
        else:
            # Fall back to inference-based attention analysis
            logger.info("Using inference-based attention analysis")
            attention_heads = self._infer_attention_behavior(
                prompt=prompt,
                output=output,
                prompt_tokens=prompt_tokens,
                output_tokens=output_tokens,
                layer_range=layer_range,
                num_heads=num_heads,
                head_threshold=head_threshold
            )
        
        # Analyze attention patterns
        head_patterns = self._analyze_attention_patterns(attention_heads)
        
        # Create attention result
        attention_result = {
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "attention_heads": attention_heads,
            "head_patterns": head_patterns,
            "metadata": {
                "prompt": prompt,
                "output": output,
                "model_id": self.model.model_id,
                "layer_range": layer_range,
                "head_threshold": head_threshold,
                "timestamp": time.time()
            }
        }
        
        # Generate visualization if requested
        if visualize and self.visualizer:
            visualization = self.visualizer.visualize_attention_heads(attention_result)
            attention_result["metadata"]["visualization"] = visualization
        
        # Record execution time
        trace_time = time.time() - trace_start
        attention_result["metadata"]["trace_time"] = trace_time
        
        logger.info(f"Attention head tracing completed in {trace_time:.2f}s")
        return attention_result
    
    def trace_qk_alignment(
        self,
        prompt: str,
        output: Optional[str] = None,
        layer_indices: Optional[List[int]] = None,
        visualize: bool = False
    ) -> Dict[str, Any]:
        """
        Trace query-key alignment in attention mechanisms.
        
        Parameters:
        -----------
        prompt : str
            Input prompt
        output : Optional[str]
            Output to trace alignment for. If None, will generate output.
        layer_indices : Optional[List[int]]
            Specific layers to analyze. If None, analyzes representative layers.
        visualize : bool
            Whether to generate visualization
            
        Returns:
        --------
        Dict[str, Any]
            Dictionary containing QK alignment analysis
        """
        trace_start = time.time()
        
        # Generate output if not provided
        if output is None:
            output = self.model.generate(prompt=prompt, max_tokens=800)
        
        # Get token-level representations
        prompt_tokens = self._tokenize(prompt)
        output_tokens = self._tokenize(output)
        
        # Get model information
        model_info = self.model.get_model_info()
        num_layers = model_info.get("num_layers", 12)
        
        # Determine layer indices
        if layer_indices is None:
            # Sample representative layers (early, middle, late)
            layer_indices = [
                0,  # First layer
                num_layers // 2,  # Middle layer
                num_layers - 1  # Last layer
            ]
        
        # Trace QK alignment
        qk_alignments = []
        
        # If direct API access is available, get QK values directly
        if hasattr(self.model, "get_qk_values"):
            try:
                logger.info("Getting QK values directly from model API")
                qk_values = self.model.get_qk_values(
                    prompt=prompt,
                    output=output,
                    layer_indices=layer_indices
                )
                qk_alignments = self._process_api_qk_values(
                    qk_values,
                    prompt_tokens,
                    output_tokens,
                    layer_indices
                )
                logger.info("Successfully processed API QK values")
            except Exception as e:
                logger.warning(f"Failed to get QK values from API: {e}")
                logger.info("Falling back to inference-based QK alignment")
                qk_alignments = self._infer_qk_alignment(
                    prompt=prompt,
                    output=output,
                    prompt_tokens=prompt_tokens,
                    output_tokens=output_tokens,
                    layer_indices=layer_indices
                )
        else:
            # Fall back to inference-based QK alignment
            logger.info("Using inference-based QK alignment")
            qk_alignments = self._infer_qk_alignment(
                prompt=prompt,
                output=output,
                prompt_tokens=prompt_tokens,
                output_tokens=output_tokens,
                layer_indices=layer_indices
            )
        
        # Analyze QK patterns
        qk_patterns = self._analyze_qk_patterns(qk_alignments)
        
        # Create QK alignment result
        qk_result = {
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "qk_alignments": qk_alignments,
            "qk_patterns": qk_patterns,
            "metadata": {
                "prompt": prompt,
                "output": output,
                "model_id": self.model.model_id,
                "layer_indices": layer_indices,
                "timestamp": time.time()
            }
        }
        
        # Generate visualization if requested
        if visualize and self.visualizer:
            visualization = self.visualizer.visualize_qk_alignment(qk_result)
            qk_result["metadata"]["visualization"] = visualization
        
        # Record execution time
        trace_time = time.time() - trace_start
        qk_result["metadata"]["trace_time"] = trace_time
        
        logger.info(f"QK alignment tracing completed in {trace_time:.2f}s")
        return qk_result
    
    def trace_ov_projection(
        self,
        prompt: str,
        output: Optional[str] = None,
        layer_indices: Optional[List[int]] = None,
        visualize: bool = False
    ) -> Dict[str, Any]:
        """
        Trace output-value projection in attention mechanisms.
        
        Parameters:
        -----------
        prompt : str
            Input prompt
        output : Optional[str]
            Output to trace projection for. If None, will generate output.
        layer_indices : Optional[List[int]]
            Specific layers to analyze. If None, analyzes representative layers.
        visualize : bool
            Whether to generate visualization
            
        Returns:
        --------
        Dict[str, Any]
            Dictionary containing OV projection analysis
        """
        trace_start = time.time()
        
        # Generate output if not provided
        if output is None:
            output = self.model.generate(prompt=prompt, max_tokens=800)
        
        # Get token-level representations
        prompt_tokens = self._tokenize(prompt)
        output_tokens = self._tokenize(output)
        
        # Get model information
        model_info = self.model.get_model_info()
        num_layers = model_info.get("num_layers", 12)
        
        # Determine layer indices
        if layer_indices is None:
            # Sample representative layers (early, middle, late)
            layer_indices = [
                0,  # First layer
                num_layers // 2,  # Middle layer
                num_layers - 1  # Last layer
            ]
        
        # Trace OV projection
        ov_projections = []
        
        # If direct API access is available, get OV values directly
        if hasattr(self.model, "get_ov_values"):
            try:
                logger.info("Getting OV values directly from model API")
                ov_values = self.model.get_ov_values(
                    prompt=prompt,
                    output=output,
                    layer_indices=layer_indices
                )
                ov_projections = self._process_api_ov_values(
                    ov_values,
                    prompt_tokens,
                    output_tokens,
                    layer_indices
                )
                logger.info("Successfully processed API OV values")
            except Exception as e:
                logger.warning(f"Failed to get OV values from API: {e}")
                logger.info("Falling back to inference-based OV projection")
                ov_projections = self._infer_ov_projection(
                    prompt=prompt,
                    output=output,
                    prompt_tokens=prompt_tokens,
                    output_tokens=output_tokens,
                    layer_indices=layer_indices
                )
        else:
            # Fall back to inference-based OV projection
            logger.info("Using inference-based OV projection")
            ov_projections = self._infer_ov_projection(
                prompt=prompt,
                output=output,
                prompt_tokens=prompt_tokens,
                output_tokens=output_tokens,
                layer_indices=layer_indices
            )
        
        # Analyze OV patterns
        ov_patterns = self._analyze_ov_patterns(ov_projections)
        
        # Create OV projection result
        ov_result = {
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "ov_projections": ov_projections,
            "ov_patterns": ov_patterns,
            "metadata": {
                "prompt": prompt,
                "output": output,
                "model_id": self.model.model_id,
                "layer_indices": layer_indices,
                "timestamp": time.time()
            }
        }
        
        # Generate visualization if requested
        if visualize and self.visualizer:
            visualization = self.visualizer.visualize_ov_projection(ov_result)
            ov_result["metadata"]["visualization"] = visualization
        
        # Record execution time
        trace_time = time.time() - trace_start
        ov_result["metadata"]["trace_time"] = trace_time
        
        logger.info(f"OV projection tracing completed in {trace_time:.2f}s")
        return ov_result
    
    def compare_attribution(
        self,
        prompt: str,
        outputs: List[str],
        include_confidence: bool = True,
        visualize: bool = False
    ) -> Dict[str, Any]:
        """
        Compare attribution across multiple outputs for the same prompt.
        
        Parameters:
        -----------
        prompt : str
            Input prompt
        outputs : List[str]
            List of outputs to compare attribution for
        include_confidence : bool
            Whether to include confidence scores
        visualize : bool
            Whether to generate visualization
            
        Returns:
        --------
        Dict[str, Any]
            Dictionary containing attribution comparison
        """
        trace_start = time.time()
        
        logger.info(f"Comparing attribution across {len(outputs)} outputs")
        
        # Get token-level representation of prompt
        prompt_tokens = self._tokenize(prompt)
        
        # Trace attribution for each output
        attribution_maps = []
        for i, output in enumerate(outputs):
            logger.info(f"Tracing attribution for output {i+1}/{len(outputs)}")
            attribution_map = self.trace(
                prompt=prompt,
                output=output,
                include_confidence=include_confidence,
                visualize=False
            )
            attribution_maps.append(attribution_map)
        
        # Compare attribution patterns
        comparison = self._compare_attribution_maps(attribution_maps)
        
        # Create comparison result
        comparison_result = {
            "prompt": prompt,
            "prompt_tokens": prompt_tokens,
            "outputs": outputs,
            "attribution_maps": attribution_maps,
            "comparison": comparison,
            "metadata": {
                "model_id": self.model.model_id,
                "num_outputs": len(outputs),
                "timestamp": time.time()
            }
        }
        
        # Generate visualization if requested
        if visualize and self.visualizer:
            visualization = self.visualizer.visualize_attribution_comparison(comparison_result)
            comparison_result["metadata"]["visualization"] = visualization
        
        # Record execution time
        trace_time = time.time() - trace_start
        comparison_result["metadata"]["trace_time"] = trace_time
        
        logger.info(f"Attribution comparison completed in {trace_time:.2f}s")
        return comparison_result
    
    def visualize_attribution(
        self,
        attribution_map: AttributionMap,
        visualization_type: str = "network",
        highlight_tokens: Optional[List[str]] = None,
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Visualize attribution map.
        
        Parameters:
        -----------
        attribution_map : AttributionMap
            Attribution map to visualize
        visualization_type : str
            Type of visualization to generate
        highlight_tokens : Optional[List[str]]
            Tokens to highlight in visualization
        output_path : Optional[str]
            Path to save visualization to
            
        Returns:
        --------
        Dict[str, Any]
            Visualization result
        """
        if self.visualizer:
            return self.visualizer.visualize_attribution(
                attribution_map=attribution_map,
                visualization_type=visualization_type,
                highlight_tokens=highlight_tokens,
                output_path=output_path
            )
        else:
            # Simple matplotlib visualization if no visualizer
            return self._simple_visualization(
                attribution_map=attribution_map,
                visualization_type=visualization_type,
                highlight_tokens=highlight_tokens,
                output_path=output_path
            )
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text using model tokenizer."""
        if hasattr(self.model, "tokenize"):
            return self.model.tokenize(text)
        else:
            # Simple whitespace tokenization fallback
            return text.split()
    
    def _infer_attribution(
        self,
        prompt: str,
        output: str,
        prompt_tokens:
