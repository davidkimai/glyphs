"""
glyph_mapper.py

Core implementation of the Glyph Mapper module for the glyphs framework.
This module transforms attribution traces, residue patterns, and attention
flows into symbolic glyph representations that visualize latent spaces.
"""

import logging
import time
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any, Set
from dataclasses import dataclass, field
import json
import hashlib
from pathlib import Path
from enum import Enum
import networkx as nx
import matplotlib.pyplot as plt
from scipy.spatial import distance
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN

from ..models.adapter import ModelAdapter
from ..attribution.tracer import AttributionMap, AttributionType, AttributionLink
from ..residue.patterns import ResiduePattern, ResidueRegistry
from ..utils.visualization_utils import VisualizationEngine

# Configure glyph-aware logging
logger = logging.getLogger("glyphs.glyph_mapper")
logger.setLevel(logging.INFO)


class GlyphType(Enum):
    """Types of glyphs for different interpretability functions."""
    ATTRIBUTION = "attribution"       # Glyphs representing attribution relations
    ATTENTION = "attention"           # Glyphs representing attention patterns
    RESIDUE = "residue"               # Glyphs representing symbolic residue
    SALIENCE = "salience"             # Glyphs representing token salience
    COLLAPSE = "collapse"             # Glyphs representing collapse patterns
    RECURSIVE = "recursive"           # Glyphs representing recursive structures
    META = "meta"                     # Glyphs representing meta-level patterns
    SENTINEL = "sentinel"             # Special marker glyphs


class GlyphSemantic(Enum):
    """Semantic dimensions captured by glyphs."""
    STRENGTH = "strength"             # Strength of the pattern
    DIRECTION = "direction"           # Directional relationship
    STABILITY = "stability"           # Stability of the pattern
    COMPLEXITY = "complexity"         # Complexity of the pattern
    RECURSION = "recursion"           # Degree of recursion
    CERTAINTY = "certainty"           # Certainty of the pattern
    TEMPORAL = "temporal"             # Temporal aspects of the pattern
    EMERGENCE = "emergence"           # Emergent properties


@dataclass
class Glyph:
    """A symbolic representation of a pattern in transformer cognition."""
    id: str                           # Unique identifier
    symbol: str                       # Unicode glyph symbol
    type: GlyphType                   # Type of glyph
    semantics: List[GlyphSemantic]    # Semantic dimensions
    position: Tuple[float, float]     # Position in 2D visualization
    size: float                       # Relative size of glyph
    color: str                        # Color of glyph
    opacity: float                    # Opacity of glyph
    source_elements: List[Any] = field(default_factory=list)  # Elements that generated this glyph
    description: Optional[str] = None  # Human-readable description
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional metadata


@dataclass
class GlyphConnection:
    """A connection between glyphs in a glyph map."""
    source_id: str                    # Source glyph ID
    target_id: str                    # Target glyph ID
    strength: float                   # Connection strength
    type: str                         # Type of connection
    directed: bool                    # Whether connection is directed
    color: str                        # Connection color
    width: float                      # Connection width
    opacity: float                    # Connection opacity
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional metadata


@dataclass
class GlyphMap:
    """A complete map of glyphs representing transformer cognition."""
    id: str                           # Unique identifier
    glyphs: List[Glyph]               # Glyphs in the map
    connections: List[GlyphConnection]  # Connections between glyphs
    source_type: str                  # Type of source data
    layout_type: str                  # Type of layout
    dimensions: Tuple[int, int]       # Dimensions of visualization
    scale: float                      # Scale factor
    focal_points: List[str] = field(default_factory=list)  # Focal glyph IDs
    regions: Dict[str, List[str]] = field(default_factory=dict)  # Named regions with glyph IDs
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional metadata


class GlyphRegistry:
    """Registry of available glyphs and their semantics."""
    
    def __init__(self):
        """Initialize the glyph registry."""
        # Attribution glyphs
        self.attribution_glyphs = {
            "direct_strong": {
                "symbol": "🔍",
                "semantics": [GlyphSemantic.STRENGTH, GlyphSemantic.CERTAINTY],
                "description": "Strong direct attribution"
            },
            "direct_medium": {
                "symbol": "🔗",
                "semantics": [GlyphSemantic.STRENGTH, GlyphSemantic.CERTAINTY],
                "description": "Medium direct attribution"
            },
            "direct_weak": {
                "symbol": "🧩",
                "semantics": [GlyphSemantic.STRENGTH, GlyphSemantic.CERTAINTY],
                "description": "Weak direct attribution"
            },
            "indirect": {
                "symbol": "⤑",
                "semantics": [GlyphSemantic.DIRECTION, GlyphSemantic.COMPLEXITY],
                "description": "Indirect attribution"
            },
            "composite": {
                "symbol": "⬥",
                "semantics": [GlyphSemantic.COMPLEXITY, GlyphSemantic.EMERGENCE],
                "description": "Composite attribution"
            },
            "fork": {
                "symbol": "🔀",
                "semantics": [GlyphSemantic.DIRECTION, GlyphSemantic.COMPLEXITY],
                "description": "Attribution fork"
            },
            "loop": {
                "symbol": "🔄",
                "semantics": [GlyphSemantic.RECURSION, GlyphSemantic.COMPLEXITY],
                "description": "Attribution loop"
            },
            "gap": {
                "symbol": "⊟",
                "semantics": [GlyphSemantic.CERTAINTY, GlyphSemantic.STABILITY],
                "description": "Attribution gap"
            }
        }
        
        # Attention glyphs
        self.attention_glyphs = {
            "focus": {
                "symbol": "🎯",
                "semantics": [GlyphSemantic.STRENGTH, GlyphSemantic.CERTAINTY],
                "description": "Attention focus point"
            },
            "diffuse": {
                "symbol": "🌫️",
                "semantics": [GlyphSemantic.STRENGTH, GlyphSemantic.CERTAINTY],
                "description": "Diffuse attention"
            },
            "induction": {
                "symbol": "📈",
                "semantics": [GlyphSemantic.TEMPORAL, GlyphSemantic.DIRECTION],
                "description": "Induction head pattern"
            },
            "inhibition": {
                "symbol": "🛑",
                "semantics": [GlyphSemantic.DIRECTION, GlyphSemantic.STRENGTH],
                "description": "Attention inhibition"
            },
            "multi_head": {
                "symbol": "⟁",
                "semantics": [GlyphSemantic.COMPLEXITY, GlyphSemantic.EMERGENCE],
                "description": "Multi-head attention pattern"
            }
        }
        
        # Residue glyphs
        self.residue_glyphs = {
            "memory_decay": {
                "symbol": "🌊",
                "semantics": [GlyphSemantic.TEMPORAL, GlyphSemantic.STABILITY],
                "description": "Memory decay residue"
            },
            "value_conflict": {
                "symbol": "⚡",
                "semantics": [GlyphSemantic.STABILITY, GlyphSemantic.CERTAINTY],
                "description": "Value conflict residue"
            },
            "ghost_activation": {
                "symbol": "👻",
                "semantics": [GlyphSemantic.STRENGTH, GlyphSemantic.CERTAINTY],
                "description": "Ghost activation residue"
            },
            "boundary_hesitation": {
                "symbol": "⧋",
                "semantics": [GlyphSemantic.CERTAINTY, GlyphSemantic.STABILITY],
                "description": "Boundary hesitation residue"
            },
            "null_output": {
                "symbol": "⊘",
                "semantics": [GlyphSemantic.CERTAINTY, GlyphSemantic.STABILITY],
                "description": "Null output residue"
            }
        }
        
        # Recursive glyphs
        self.recursive_glyphs = {
            "recursive_aegis": {
                "symbol": "🜏",
                "semantics": [GlyphSemantic.RECURSION, GlyphSemantic.STABILITY],
                "description": "Recursive immunity"
            },
            "recursive_seed": {
                "symbol": "∴",
                "semantics": [GlyphSemantic.RECURSION, GlyphSemantic.EMERGENCE],
                "description": "Recursion initiation"
            },
            "recursive_exchange": {
                "symbol": "⇌",
                "semantics": [GlyphSemantic.RECURSION, GlyphSemantic.DIRECTION],
                "description": "Bidirectional recursion"
            },
            "recursive_mirror": {
                "symbol": "🝚",
                "semantics": [GlyphSemantic.RECURSION, GlyphSemantic.EMERGENCE],
                "description": "Recursive reflection"
            },
            "recursive_anchor": {
                "symbol": "☍",
                "semantics": [GlyphSemantic.RECURSION, GlyphSemantic.STABILITY],
                "description": "Stable recursive reference"
            }
        }
        
        # Meta glyphs
        self.meta_glyphs = {
            "uncertainty": {
                "symbol": "❓",
                "semantics": [GlyphSemantic.CERTAINTY],
                "description": "Uncertainty marker"
            },
            "emergence": {
                "symbol": "✧",
                "semantics": [GlyphSemantic.EMERGENCE, GlyphSemantic.COMPLEXITY],
                "description": "Emergent pattern marker"
            },
            "collapse_point": {
                "symbol": "💥",
                "semantics": [GlyphSemantic.STABILITY, GlyphSemantic.CERTAINTY],
                "description": "Collapse point marker"
            },
            "temporal_marker": {
                "symbol": "⧖",
                "semantics": [GlyphSemantic.TEMPORAL],
                "description": "Temporal sequence marker"
            }
        }
        
        # Sentinel glyphs
        self.sentinel_glyphs = {
            "start": {
                "symbol": "◉",
                "semantics": [GlyphSemantic.DIRECTION],
                "description": "Start marker"
            },
            "end": {
                "symbol": "◯",
                "semantics": [GlyphSemantic.DIRECTION],
                "description": "End marker"
            },
            "boundary": {
                "symbol": "⬚",
                "semantics": [GlyphSemantic.STABILITY],
                "description": "Boundary marker"
            },
            "reference": {
                "symbol": "✱",
                "semantics": [GlyphSemantic.DIRECTION],
                "description": "Reference marker"
            }
        }
        
        # Combine all glyphs into a single map
        self.all_glyphs = {
            **{f"attribution_{k}": v for k, v in self.attribution_glyphs.items()},
            **{f"attention_{k}": v for k, v in self.attention_glyphs.items()},
            **{f"residue_{k}": v for k, v in self.residue_glyphs.items()},
            **{f"recursive_{k}": v for k, v in self.recursive_glyphs.items()},
            **{f"meta_{k}": v for k, v in self.meta_glyphs.items()},
            **{f"sentinel_{k}": v for k, v in self.sentinel_glyphs.items()}
        }
    
    def get_glyph(self, glyph_id: str) -> Dict[str, Any]:
        """Get a glyph by ID."""
        if glyph_id in self.all_glyphs:
            return self.all_glyphs[glyph_id]
        else:
            raise ValueError(f"Unknown glyph ID: {glyph_id}")
    
    def find_glyphs_by_semantic(self, semantic: GlyphSemantic) -> List[str]:
        """Find glyphs that have a specific semantic dimension."""
        return [
            glyph_id for glyph_id, glyph in self.all_glyphs.items()
            if semantic in glyph.get("semantics", [])
        ]
    
    def find_glyphs_by_type(self, glyph_type: str) -> List[str]:
        """Find glyphs of a specific type."""
        return [
            glyph_id for glyph_id in self.all_glyphs.keys()
            if glyph_id.startswith(f"{glyph_type}_")
        ]


class GlyphMapper:
    """
    Core glyph mapping system for the glyphs framework.
    
    This class transforms attribution traces, residue patterns, and attention
    flows into symbolic glyph representations that visualize latent spaces.
    It serves as the bridge between raw interpretability data and meaningful
    symbolic visualization.
    """
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        visualizer: Optional[VisualizationEngine] = None
    ):
        """
        Initialize the glyph mapper.
        
        Parameters:
        -----------
        config : Optional[Dict[str, Any]]
            Configuration parameters for the mapper
        visualizer : Optional[VisualizationEngine]
            Visualization engine for glyph visualization
        """
        self.config = config or {}
        self.visualizer = visualizer
        
        # Configure mapper parameters
        self.min_connection_strength = self.config.get("min_connection_strength", 0.1)
        self.auto_layout = self.config.get("auto_layout", True)
        self.default_layout = self.config.get("default_layout", "force_directed")
        self.default_dimensions = self.config.get("default_dimensions", (800, 600))
        self.default_scale = self.config.get("default_scale", 1.0)
        self.connection_bundling = self.config.get("connection_bundling", True)
        self.color_scheme = self.config.get("color_scheme", "semantic")
        
        # Initialize glyph registry
        self.registry = GlyphRegistry()
        
        # Track created glyph maps
        self.glyph_map_history = []
        
        logger.info("Glyph mapper initialized")
    
    def map_attribution(
        self,
        attribution_map: AttributionMap,
        layout_type: Optional[str] = None,
        dimensions: Optional[Tuple[int, int]] = None,
        scale: Optional[float] = None,
        include_tokens: bool = True,
        focus_on: Optional[List[str]] = None
    ) -> GlyphMap:
        """
        Map attribution patterns to glyphs.
        
        Parameters:
        -----------
        attribution_map : AttributionMap
            Attribution map to visualize
        layout_type : Optional[str]
            Type of layout to use
        dimensions : Optional[Tuple[int, int]]
            Dimensions for visualization
        scale : Optional[float]
            Scale factor for visualization
        include_tokens : bool
            Whether to include tokens as sentinel glyphs
        focus_on : Optional[List[str]]
            Tokens to focus visualization on
            
        Returns:
        --------
        GlyphMap
            Glyph map representation of attribution
        """
        map_start = time.time()
        layout_type = layout_type or self.default_layout
        dimensions = dimensions or self.default_dimensions
        scale = scale or self.default_scale
        
        logger.info(f"Mapping attribution to glyphs with {layout_type} layout")
        
        # Create unique ID for glyph map
        map_id = f"attribution_{int(time.time())}_{hashlib.md5(str(attribution_map).encode()).hexdigest()[:8]}"
        
        # Initialize glyph map
        glyph_map = GlyphMap(
            id=map_id,
            glyphs=[],
            connections=[],
            source_type="attribution",
            layout_type=layout_type,
            dimensions=dimensions,
            scale=scale,
            metadata={
                "attribution_map_id": attribution_map.metadata.get("id", "unknown"),
                "prompt": attribution_map.metadata.get("prompt", ""),
                "output": attribution_map.metadata.get("output", ""),
                "model_id": attribution_map.metadata.get("model_id", "unknown"),
                "timestamp": time.time()
            }
        )
        
        # Add sentinel glyphs for tokens if requested
        token_glyph_ids = {}
        if include_tokens:
            # Add prompt tokens
            for i, token in enumerate(attribution_map.prompt_tokens):
                glyph_id = f"token_prompt_{i}"
                glyph = Glyph(
                    id=glyph_id,
                    symbol="◆",  # Diamond for prompt tokens
                    type=GlyphType.SENTINEL,
                    semantics=[GlyphSemantic.DIRECTION],
                    position=(0, i * 20),  # Initial position, will be updated by layout
                    size=5.0,
                    color="#3498db",  # Blue
                    opacity=0.8,
                    source_elements=[token],
                    description=f"Prompt token: '{token}'",
                    metadata={"token_index": i, "token_type": "prompt"}
                )
                glyph_map.glyphs.append(glyph)
                token_glyph_ids[f"prompt_{i}"] = glyph_id
            
            # Add output tokens
            for i, token in enumerate(attribution_map.output_tokens):
                glyph_id = f"token_output_{i}"
                glyph = Glyph(
                    id=glyph_id,
                    symbol="◇",  # Open diamond for output tokens
                    type=GlyphType.SENTINEL,
                    semantics=[GlyphSemantic.DIRECTION],
                    position=(100, i * 20),  # Initial position, will be updated by layout
                    size=5.0,
                    color="#e74c3c",  # Red
                    opacity=0.8,
                    source_elements=[token],
                    description=f"Output token: '{token}'",
                    metadata={"token_index": i, "token_type": "output"}
                )
                glyph_map.glyphs.append(glyph)
                token_glyph_ids[f"output_{i}"] = glyph_id
        
        # Process attribution links
        link_glyphs = {}
        for link_idx, link in enumerate(attribution_map.links):
            # Skip weak attributions
            if link.strength < self.min_connection_strength:
                continue
            
            # Determine glyph type based on attribution type and strength
            if link.attribution_type == AttributionType.DIRECT:
                if link.strength > 0.7:
                    glyph_type = "attribution_direct_strong"
                elif link.strength > 0.4:
                    glyph_type = "attribution_direct_medium"
                else:
                    glyph_type = "attribution_direct_weak"
            elif link.attribution_type == AttributionType.INDIRECT:
                glyph_type = "attribution_indirect"
            elif link.attribution_type == AttributionType.COMPOSITE:
                glyph_type = "attribution_composite"
            elif link.attribution_type == AttributionType.RECURSIVE:
                glyph_type = "recursive_recursive_exchange"
            else:
                glyph_type = "attribution_direct_weak"
            
            # Get glyph from registry
            glyph_info = self.registry.get_glyph(glyph_type)
            
            # Create glyph for this attribution link
            glyph_id = f"link_{link_idx}"
            glyph = Glyph(
                id=glyph_id,
                symbol=glyph_info["symbol"],
                type=GlyphType.ATTRIBUTION,
                semantics=glyph_info["semantics"],
                position=(50, (link.source_idx + link.target_idx) * 10),  # Initial position
                size=10.0 * link.strength,  # Size based on strength
                color=self._get_color_for_attribution(link),
                opacity=min(1.0, 0.5 + link.strength / 2),
                source_elements=[link],
                description=glyph_info["description"],
                metadata={
                    "link_index": link_idx,
                    "source_index": link.source_idx,
                    "target_index": link.target_idx,
                    "attribution_type": str(link.attribution_type),
                    "strength": link.strength
                }
            )
            glyph_map.glyphs.append(glyph)
            link_glyphs[link_idx] = glyph_id
            
            # Create connections to token glyphs if tokens are included
            if include_tokens:
                source_glyph_id = token_glyph_ids.get(f"prompt_{link.source_idx}")
                target_glyph_id = token_glyph_ids.get(f"output_{link.target_idx}")
                
                if source_glyph_id and target_glyph_id:
                    # Connection from source token to attribution glyph
                    glyph_map.connections.append(GlyphConnection(
                        source_id=source_glyph_id,
                        target_id=glyph_id,
                        strength=link.strength,
                        type="attribution_flow",
                        directed=True,
                        color="#7f8c8d",  # Gray
                        width=1.0 + 2.0 * link.strength,
                        opacity=0.7 * link.strength
                    ))
                    
                    # Connection from attribution glyph to target token
                    glyph_map.connections.append(GlyphConnection(
                        source_id=glyph_id,
                        target_id=target_glyph_id,
                        strength=link.strength,
                        type="attribution_flow",
                        directed=True,
                        color="#7f8c8d",  # Gray
                        width=1.0 + 2.0 * link.strength,
                        opacity=0.7 * link.strength
                    ))
        
        # Process attribution gaps
        for gap_idx, (start_idx, end_idx) in enumerate(attribution_map.attribution_gaps):
            # Add attribution gap glyph
            glyph_info = self.registry.get_glyph("attribution_gap")
            glyph_id = f"gap_{gap_idx}"
            glyph = Glyph(
                id=glyph_id,
                symbol=glyph_info["symbol"],
                type=GlyphType.ATTRIBUTION,
                semantics=glyph_info["semantics"],
                position=(50, (start_idx + end_idx) * 10),  # Initial position
                size=8.0,
                color="#e67e22",  # Orange
                opacity=0.8,
                source_elements=[(start_idx, end_idx)],
                description=f"Attribution gap between indices {start_idx} and {end_idx}",
                metadata={
                    "gap_index": gap_idx,
                    "start_index": start_idx,
                    "end_index": end_idx
                }
            )
            glyph_map.glyphs.append(glyph)
            
            # Create connections to token glyphs if tokens are included
            if include_tokens:
                source_glyph_id = token_glyph_ids.get(f"prompt_{start_idx}")
                target_glyph_id = token_glyph_ids.get(f"output_{end_idx}")
                
                if source_glyph_id and target_glyph_id:
                    # Connect source token to gap glyph
                    glyph_map.connections.append(GlyphConnection(
                        source_id=source_glyph_id,
                        target_id=glyph_id,
                        strength=0.5,
                        type="attribution_gap",
                        directed=True,
                        color="#e67e22",  # Orange
                        width=1.5,
                        opacity=0.6
                    ))
                    
                    # Connect gap glyph to target token
                    glyph_map.connections.append(GlyphConnection(
                        source_id=glyph_id,
                        target_id=target_glyph_id,
                        strength=0.5,
                        type="attribution_gap",
                        directed=True,
                        color="#e67e22",  # Orange
                        width=1.5,
                        opacity=0.6
                    ))
        
        # Process collapsed regions
        for collapse_idx, (start_idx, end_idx) in enumerate(attribution_map.collapsed_regions):
            # Add collapse glyph
            glyph_info = self.registry.get_glyph("meta_collapse_point")
            glyph_id = f"collapse_{collapse_idx}"
            glyph = Glyph(
                id=glyph_id,
                symbol=glyph_info["symbol"],
                type=GlyphType.COLLAPSE,
                semantics=glyph_info["semantics"],
                position=(50, (start_idx + end_idx) * 10),  # Initial position
                size=12.0,
                color="#9b59b6",  # Purple
                opacity=0.9,
                source_elements=[(start_idx, end_idx)],
                description=f"Attribution collapse between indices {start_idx} and {end_idx}",
                metadata={
                    "collapse_index": collapse_idx,
                    "start_index": start_idx,
                    "end_index": end_idx
                }
            )
            glyph_map.glyphs.append(glyph)
            
            # Create connections to token glyphs if tokens are included
            if include_tokens:
                # Connect to all tokens in the collapsed region
                for i in range(start_idx, end_idx + 1):
                    token_glyph_id = token_glyph_ids.get(f"output_{i}")
                    if token_glyph_id:
                        glyph_map.connections.append(GlyphConnection(
                            source_id=glyph_id,
                            target_id=token_glyph_id,
                            strength=0.7,
                            type="collapse_effect",
                            directed=True,
                            color="#9b59b6",  # Purple
                            width=1.5,
                            opacity=0.7
                        ))
        
        # Identify focal points based on token salience
        if attribution_map.token_salience:
            # Find top salient tokens
            salient_tokens = sorted(
                attribution_map.token_salience.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]  # Top 5 salient tokens
            
            for token_idx, salience in salient_tokens:
                # Add to focal points if salience is significant
                if salience > 0.5:
                    token_glyph_id = token_glyph_ids.get(f"output_{token_idx}")
                    if token_glyph_id:
                        glyph_map.focal_points.append(token_glyph_id)
                    
                    # Add salience glyph for highly salient tokens
                    if salience > 0.8:
                        glyph_info = self.registry.get_glyph("attention_focus")
                        glyph_id = f"salience_{token_idx}"
                        glyph = Glyph(
                            id=glyph_id,
                            symbol=glyph_info["symbol"],
                            type=GlyphType.SALIENCE,
                            semantics=glyph_info["semantics"],
                            position=(120, token_idx * 20),  # Initial position
                            size=10.0 * salience,
                            color="#f1c40f",  # Yellow
                            opacity=salience,
                            source_elements=[token_idx],
                            description=f"High salience token at index {token_idx}",
                            metadata={
                                "token_index": token_idx,
                                "salience": salience
                            }
                        )
                        glyph_map.glyphs.append(glyph)
                        
                        # Connect salience glyph to token
                        glyph_map.connections.append(GlyphConnection(
                            source_id=glyph_id,
                            target_id=token_glyph_id,
                            strength=salience,
                            type="salience_marker",
                            directed=False,
                            color="#f1c40f",  # Yellow
                            width=2.0 * salience,
                            opacity=0.8
                        ))
        
        # Apply layout if auto_layout is enabled
        if self.auto_layout:
            glyph_map = self._apply_layout(glyph_map, layout_type)
        
        # Apply focus if specified
        if focus_on:
            glyph_map = self._apply_focus(glyph_map, focus_on)
        
        # Record execution time
        map_time = time.time() - map_start
        glyph_map.metadata["map_time"] = map_time
        
        # Add to history
        self.glyph_map_history.append(glyph_map)
        
        logger.info(f"Attribution mapping completed in {map_time:.2f}s")
        return glyph_map
    
    def map_residue_patterns(
        self,
        residue_patterns: List[ResiduePattern],
        layout_type: Optional[str] = None,
        dimensions: Optional[Tuple[int, int]] = None,
        scale: Optional[float] = None,
        cluster_patterns: bool = True
    ) ->
