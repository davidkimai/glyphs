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
    def map_residue_patterns(
        self,
        residue_patterns: List[ResiduePattern],
        layout_type: Optional[str] = None,
        dimensions: Optional[Tuple[int, int]] = None,
        scale: Optional[float] = None,
        cluster_patterns: bool = True
    ) -> GlyphMap:
        """
        Map residue patterns to glyphs.
        
        Parameters:
        -----------
        residue_patterns : List[ResiduePattern]
            Residue patterns to visualize
        layout_type : Optional[str]
            Type of layout to use
        dimensions : Optional[Tuple[int, int]]
            Dimensions for visualization
        scale : Optional[float]
            Scale factor for visualization
        cluster_patterns : bool
            Whether to cluster similar patterns
            
        Returns:
        --------
        GlyphMap
            Glyph map representation of residue patterns
        """
        map_start = time.time()
        layout_type = layout_type or self.default_layout
        dimensions = dimensions or self.default_dimensions
        scale = scale or self.default_scale
        
        logger.info(f"Mapping {len(residue_patterns)} residue patterns to glyphs")
        
        # Create unique ID for glyph map
        map_id = f"residue_{int(time.time())}_{hashlib.md5(str(residue_patterns).encode()).hexdigest()[:8]}"
        
        # Initialize glyph map
        glyph_map = GlyphMap(
            id=map_id,
            glyphs=[],
            connections=[],
            source_type="residue",
            layout_type=layout_type,
            dimensions=dimensions,
            scale=scale,
            metadata={
                "num_patterns": len(residue_patterns),
                "pattern_types": [p.type for p in residue_patterns],
                "timestamp": time.time()
            }
        )
        
        # Group patterns by type
        pattern_by_type = {}
        for pattern in residue_patterns:
            if pattern.type not in pattern_by_type:
                pattern_by_type[pattern.type] = []
            pattern_by_type[pattern.type].append(pattern)
        
        # Create type center glyphs
        type_glyphs = {}
        for i, (pattern_type, patterns) in enumerate(pattern_by_type.items()):
            # Determine glyph based on pattern type
            if pattern_type == "memory_decay":
                glyph_type = "residue_memory_decay"
            elif pattern_type == "value_conflict":
                glyph_type = "residue_value_conflict"
            elif pattern_type == "ghost_activation":
                glyph_type = "residue_ghost_activation"
            elif pattern_type == "boundary_hesitation":
                glyph_type = "residue_boundary_hesitation"
            elif pattern_type == "null_output":
                glyph_type = "residue_null_output"
            else:
                # Default to null output for unknown types
                glyph_type = "residue_null_output"
            
            # Get glyph from registry
            glyph_info = self.registry.get_glyph(glyph_type)
            
            # Create center glyph for this pattern type
            glyph_id = f"type_{pattern_type}"
            glyph = Glyph(
                id=glyph_id,
                symbol=glyph_info["symbol"],
                type=GlyphType.RESIDUE,
                semantics=glyph_info["semantics"],
                position=(dimensions[0] / 2, 100 + i * 150),  # Initial position
                size=20.0,  # Larger for type centers
                color=self._get_color_for_residue_type(pattern_type),
                opacity=1.0,
                source_elements=patterns,
                description=f"{pattern_type.replace('_', ' ').title()} Residue Pattern",
                metadata={
                    "pattern_type": pattern_type,
                    "pattern_count": len(patterns),
                    "average_confidence": sum(p.confidence for p in patterns) / len(patterns)
                }
            )
            glyph_map.glyphs.append(glyph)
            type_glyphs[pattern_type] = glyph_id
            
            # Add to regions
            glyph_map.regions[pattern_type] = [glyph_id]
        
        # For each pattern, create instance glyph
        instance_glyphs = {}
        for pattern_idx, pattern in enumerate(residue_patterns):
            # Get parent type glyph
            parent_glyph_id = type_glyphs.get(pattern.type)
            if not parent_glyph_id:
                continue
            
            # Determine specific glyph variant based on confidence
            if pattern.confidence > 0.8:
                size_factor = 1.2
                opacity = 0.9
            elif pattern.confidence > 0.5:
                size_factor = 1.0
                opacity = 0.7
            else:
                size_factor = 0.8
                opacity = 0.5
            
            # Get glyph from registry
            glyph_type = f"residue_{pattern.type}"
            glyph_info = self.registry.get_glyph(glyph_type)
            
            # Create glyph for this pattern instance
            glyph_id = f"pattern_{pattern_idx}"
            glyph = Glyph(
                id=glyph_id,
                symbol=glyph_info["symbol"],
                type=GlyphType.RESIDUE,
                semantics=glyph_info["semantics"],
                position=(0, 0),  # Will be set by layout
                size=12.0 * size_factor,
                color=self._get_color_for_residue_type(pattern.type),
                opacity=opacity,
                source_elements=[pattern],
                description=f"{pattern.type.replace('_', ' ').title()} Pattern: {pattern.signature[:20]}",
                metadata={
                    "pattern_type": pattern.type,
                    "confidence": pattern.confidence,
                    "signature": pattern.signature,
                    "context": pattern.context
                }
            )
            glyph_map.glyphs.append(glyph)
            instance_glyphs[pattern_idx] = glyph_id
            
            # Add to regions
            if pattern.type in glyph_map.regions:
                glyph_map.regions[pattern.type].append(glyph_id)
            
            # Connect to parent type glyph
            glyph_map.connections.append(GlyphConnection(
                source_id=parent_glyph_id,
                target_id=glyph_id,
                strength=pattern.confidence,
                type="type_instance",
                directed=True,
                color=self._get_color_for_residue_type(pattern.type),
                width=1.0 + pattern.confidence,
                opacity=0.6 * pattern.confidence
            ))
        
        # Connect similar patterns if clustering enabled
        if cluster_patterns and len(residue_patterns) > 1:
            # Create similarity matrix based on pattern signatures
            similarity_matrix = np.zeros((len(residue_patterns), len(residue_patterns)))
            
            for i, pattern1 in enumerate(residue_patterns):
                for j, pattern2 in enumerate(residue_patterns):
                    if i == j:
                        similarity_matrix[i, j] = 1.0
                    else:
                        # Calculate similarity based on signature and context
                        signature_sim = self._calculate_signature_similarity(
                            pattern1.signature, pattern2.signature
                        )
                        context_sim = self._calculate_context_similarity(
                            pattern1.context, pattern2.context
                        )
                        # Weighted combination
                        similarity_matrix[i, j] = 0.7 * signature_sim + 0.3 * context_sim
            
            # Find significant connections
            for i in range(len(residue_patterns)):
                for j in range(i + 1, len(residue_patterns)):
                    similarity = similarity_matrix[i, j]
                    if similarity > 0.6:  # Only connect sufficiently similar patterns
                        source_id = instance_glyphs.get(i)
                        target_id = instance_glyphs.get(j)
                        if source_id and target_id:
                            glyph_map.connections.append(GlyphConnection(
                                source_id=source_id,
                                target_id=target_id,
                                strength=similarity,
                                type="pattern_similarity",
                                directed=False,
                                color="#2ecc71",  # Green
                                width=1.0 + 2.0 * similarity,
                                opacity=0.5 * similarity
                            ))
            
            # Use similarity matrix for layout
            if layout_type == "similarity":
                # Use t-SNE to layout based on similarity
                tsne = TSNE(n_components=2, perplexity=min(5, len(residue_patterns) - 1))
                positions = tsne.fit_transform(similarity_matrix)
                
                # Scale positions to dimensions
                positions = self._scale_positions(positions, dimensions)
                
                # Update glyph positions for pattern instances
                for i, pattern_idx in enumerate(instance_glyphs.keys()):
                    glyph_id = instance_glyphs[pattern_idx]
                    for glyph in glyph_map.glyphs:
                        if glyph.id == glyph_id:
                            glyph.position = (positions[i, 0], positions[i, 1])
        
        # Apply layout if auto_layout is enabled and not already layout by similarity
        if self.auto_layout and layout_type != "similarity":
            glyph_map = self._apply_layout(glyph_map, layout_type)
        
        # Add sentinel glyphs for context markers
        for pattern_idx, pattern in enumerate(residue_patterns):
            # Only add context markers for high-confidence patterns
            if pattern.confidence < 0.7:
                continue
            
            glyph_id = instance_glyphs.get(pattern_idx)
            if not glyph_id:
                continue
            
            # Add reference glyph for context
            ref_glyph_id = f"ref_{pattern_idx}"
            ref_glyph = Glyph(
                id=ref_glyph_id,
                symbol="✱",  # Star for reference
                type=GlyphType.SENTINEL,
                semantics=[GlyphSemantic.DIRECTION],
                position=(0, 0),  # Will be set relative to pattern glyph
                size=6.0,
                color="#3498db",  # Blue
                opacity=0.7,
                source_elements=[pattern.context],
                description=f"Context reference for pattern {pattern_idx}",
                metadata={
                    "pattern_index": pattern_idx,
                    "reference_type": "context"
                }
            )
            
            # Position relative to pattern glyph
            for glyph in glyph_map.glyphs:
                if glyph.id == glyph_id:
                    x, y = glyph.position
                    ref_glyph.position = (x + 20, y - 20)
            
            glyph_map.glyphs.append(ref_glyph)
            
            # Connect reference to pattern
            glyph_map.connections.append(GlyphConnection(
                source_id=ref_glyph_id,
                target_id=glyph_id,
                strength=0.7,
                type="context_reference",
                directed=True,
                color="#3498db",  # Blue
                width=1.0,
                opacity=0.5
            ))
        
        # Record execution time
        map_time = time.time() - map_start
        glyph_map.metadata["map_time"] = map_time
        
        # Add to history
        self.glyph_map_history.append(glyph_map)
        
        logger.info(f"Residue pattern mapping completed in {map_time:.2f}s")
        return glyph_map
    
    def map_attention_heads(
        self,
        attention_data: Dict[str, Any],
        layout_type: Optional[str] = None,
        dimensions: Optional[Tuple[int, int]] = None,
        scale: Optional[float] = None,
        include_tokens: bool = True
    ) -> GlyphMap:
        """
        Map attention head patterns to glyphs.
        
        Parameters:
        -----------
        attention_data : Dict[str, Any]
            Attention head data to visualize
        layout_type : Optional[str]
            Type of layout to use
        dimensions : Optional[Tuple[int, int]]
            Dimensions for visualization
        scale : Optional[float]
            Scale factor for visualization
        include_tokens : bool
            Whether to include tokens as sentinel glyphs
            
        Returns:
        --------
        GlyphMap
            Glyph map representation of attention head patterns
        """
        map_start = time.time()
        layout_type = layout_type or self.default_layout
        dimensions = dimensions or self.default_dimensions
        scale = scale or self.default_scale
        
        logger.info("Mapping attention head patterns to glyphs")
        
        # Extract data
        prompt_tokens = attention_data.get("prompt_tokens", [])
        output_tokens = attention_data.get("output_tokens", [])
        attention_heads = attention_data.get("attention_heads", [])
        head_patterns = attention_data.get("head_patterns", {})
        
        # Create unique ID for glyph map
        map_id = f"attention_{int(time.time())}_{hashlib.md5(str(attention_heads).encode()).hexdigest()[:8]}"
        
        # Initialize glyph map
        glyph_map = GlyphMap(
            id=map_id,
            glyphs=[],
            connections=[],
            source_type="attention",
            layout_type=layout_type,
            dimensions=dimensions,
            scale=scale,
            metadata={
                "num_heads": len(attention_heads),
                "model_id": attention_data.get("metadata", {}).get("model_id", "unknown"),
                "timestamp": time.time()
            }
        )
        
        # Add sentinel glyphs for tokens if requested
        token_glyph_ids = {}
        if include_tokens:
            # Add prompt tokens
            for i, token in enumerate(prompt_tokens):
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
            for i, token in enumerate(output_tokens):
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
        
        # Process attention heads
        head_glyphs = {}
        for head_idx, head in enumerate(attention_heads):
            # Determine glyph type based on attention pattern
            if head.pattern_type == "induction":
                glyph_type = "attention_induction"
            elif head.pattern_type == "focus":
                glyph_type = "attention_focus"
            elif head.pattern_type == "diffuse":
                glyph_type = "attention_diffuse"
            elif head.pattern_type == "inhibition":
                glyph_type = "attention_inhibition"
            else:
                glyph_type = "attention_multi_head"
            
            # Get glyph from registry
            glyph_info = self.registry.get_glyph(glyph_type)
            
            # Create glyph for this attention head
            glyph_id = f"head_{head_idx}"
            glyph = Glyph(
                id=glyph_id,
                symbol=glyph_info["symbol"],
                type=GlyphType.ATTENTION,
                semantics=glyph_info["semantics"],
                position=(50, head.layer * 40 + head.head * 10),  # Initial position
                size=8.0 + 5.0 * head.strength,  # Size based on strength
                color=self._get_color_for_layer(head.layer),
                opacity=min(1.0, 0.5 + head.strength / 2),
                source_elements=[head],
                description=f"Attention head {head.layer}.{head.head}: {head.pattern_type}",
                metadata={
                    "head_index": head_idx,
                    "layer": head.layer,
                    "head": head.head,
                    "pattern_type": head.pattern_type,
                    "strength": head.strength,
                    "function": head.function,
                    "attribution_role": head.attribution_role
                }
            )
            glyph_map.glyphs.append(glyph)
            head_glyphs[head_idx] = glyph_id
            
            # Connect to focus tokens if include_tokens is True
            if include_tokens and head.focus_tokens:
                for token_idx in head.focus_tokens:
                    token_type = "prompt" if token_idx < len(prompt_tokens) else "output"
                    adjusted_idx = token_idx if token_type == "prompt" else token_idx - len(prompt_tokens)
                    token_glyph_id = token_glyph_ids.get(f"{token_type}_{adjusted_idx}")
                    if token_glyph_id:
                        glyph_map.connections.append(GlyphConnection(
                            source_id=glyph_id,
                            target_id=token_glyph_id,
                            strength=head.strength,
                            type="attention_focus",
                            directed=True,
                            color=self._get_color_for_layer(head.layer, alpha=0.6),
                            width=1.0 + 2.0 * head.strength,
                            opacity=0.6 * head.strength
                        ))
        
        # Add pattern connections between heads
        for pattern_name, related_heads in head_patterns.items():
            # Skip patterns with less than 2 heads
            if len(related_heads) < 2:
                continue
            
            # Create pattern node
            glyph_type = "meta_emergence" if "emergent" in pattern_name else "recursive_recursive_exchange"
            glyph_info = self.registry.get_glyph(glyph_type)
            
            pattern_glyph_id = f"pattern_{pattern_name}"
            pattern_glyph = Glyph(
                id=pattern_glyph_id,
                symbol=glyph_info["symbol"],
                type=GlyphType.META if "emergent" in pattern_name else GlyphType.RECURSIVE,
                semantics=glyph_info["semantics"],
                position=(100, 100),  # Will be updated by layout
                size=12.0,
                color="#f1c40f",  # Yellow
                opacity=0.9,
                source_elements=[pattern_name, related_heads],
                description=f"Attention pattern: {pattern_name}",
                metadata={
                    "pattern_name": pattern_name,
                    "related_heads": related_heads
                }
            )
            glyph_map.glyphs.append(pattern_glyph)
            
            # Connect pattern to all related heads
            for head_idx in related_heads:
                head_glyph_id = head_glyphs.get(head_idx)
                if head_glyph_id:
                    glyph_map.connections.append(GlyphConnection(
                        source_id=pattern_glyph_id,
                        target_id=head_glyph_id,
                        strength=0.8,
                        type="pattern_membership",
                        directed=True,
                        color="#f1c40f",  # Yellow
                        width=1.5,
                        opacity=0.7
                    ))
            
            # Add pattern to regions
            if pattern_name not in glyph_map.regions:
                glyph_map.regions[pattern_name] = []
            glyph_map.regions[pattern_name].append(pattern_glyph_id)
            for head_idx in related_heads:
                head_glyph_id = head_glyphs.get(head_idx)
                if head_glyph_id:
                    glyph_map.regions[pattern_name].append(head_glyph_id)
        
        # Add layer groups
        layers = set(head.layer for head in attention_heads)
        for layer in layers:
            # Create layer group
            layer_heads = [
                head_idx for head_idx, head in enumerate(attention_heads)
                if head.layer == layer
            ]
            
            # Add layer to regions
            layer_region = f"layer_{layer}"
            glyph_map.regions[layer_region] = [
                head_glyphs[head_idx] for head_idx in layer_heads
                if head_idx in head_glyphs
            ]
        
        # Apply layout if auto_layout is enabled
        if self.auto_layout:
            glyph_map = self._apply_layout(glyph_map, layout_type)
        
        # Record execution time
        map_time = time.time() - map_start
        glyph_map.metadata["map_time"] = map_time
        
        # Add to history
        self.glyph_map_history.append(glyph_map)
        
        logger.info(f"Attention head mapping completed in {map_time:.2f}s")
        return glyph_map
    
    def map_recursive_trace(
        self,
        trace_data: Dict[str, Any],
        layout_type: Optional[str] = None,
        dimensions: Optional[Tuple[int, int]] = None,
        scale: Optional[float] = None,
        depth_limit: Optional[int] = None
    ) -> GlyphMap:
        """
        Map recursive trace data to glyphs.
        
        Parameters:
        -----------
        trace_data : Dict[str, Any]
            Recursive trace data to visualize
        layout_type : Optional[str]
            Type of layout to use
        dimensions : Optional[Tuple[int, int]]
            Dimensions for visualization
        scale : Optional[float]
            Scale factor for visualization
        depth_limit : Optional[int]
            Maximum recursion depth to visualize
            
        Returns:
        --------
        GlyphMap
            Glyph map representation of recursive trace
        """
        map_start = time.time()
        layout_type = layout_type or self.default_layout
        dimensions = dimensions or self.default_dimensions
        scale = scale or self.default_scale
        
        logger.info("Mapping recursive trace to glyphs")
        
        # Extract trace data
        trace_operations = trace_data.get("operations", [])
        trace_result = trace_data.get("result", {})
        trace_depth = trace_data.get("depth", 0)
        
        # Apply depth limit if specified
        if depth_limit is not None:
            trace_depth = min(trace_depth, depth_limit)
            # Filter operations by depth
            trace_operations = [
                op for op in trace_operations
                if op.get("depth", 0) <= depth_limit
            ]
        
        # Create unique ID for glyph map
        map_id = f"recursive_{int(time.time())}_{hashlib.md5(str(trace_data).encode()).hexdigest()[:8]}"
        
        # Initialize glyph map
        glyph_map = GlyphMap(
            id=map_id,
            glyphs=[],
            connections=[],
            source_type="recursive",
            layout_type=layout_type,
            dimensions=dimensions,
            scale=scale,
            metadata={
                "trace_command": trace_data.get("command", "unknown"),
                "trace_target": trace_data.get("target", "unknown"),
                "trace_depth": trace_depth,
                "timestamp": time.time()
            }
        )
        
        # Add recursive seed glyph
        seed_glyph = Glyph(
            id="seed",
            symbol="∴",  # Recursive seed symbol
            type=GlyphType.RECURSIVE,
            semantics=[GlyphSemantic.RECURSION, GlyphSemantic.EMERGENCE],
            position=(dimensions[0] / 2, 50),  # Top center
            size=15.0,
            color="#9b59b6",  # Purple
            opacity=1.0,
            source_elements=[trace_data.get("command", "")],
            description=f"Recursive seed: {trace_data.get('command', 'unknown')}",
            metadata={
                "command": trace_data.get("command", ""),
                "target": trace_data.get("target", ""),
                "depth": trace_depth
            }
        )
        glyph_map.glyphs.append(seed_glyph)
        glyph_map.focal_points.append("seed")
        
        # Process operations by depth
        depth_glyphs = {"0": "seed"}
        for op_idx, operation in enumerate(trace_operations):
            op_depth = operation.get("depth", 0)
            op_type = operation.get("type", "unknown")
            
            # Determine glyph type based on operation type
            if "reflect" in op_type:
                glyph_type = "recursive_recursive_mirror"
            elif "collapse" in op_type:
                glyph_type = "meta_collapse_point"
            elif "fork" in op_type:
                glyph_type = "attribution_fork"
            elif "trace" in op_type:
                glyph_type = "recursive_recursive_exchange"
            else:
                glyph_type = "recursive_recursive_anchor"
            
            # Get glyph from registry
            glyph_info = self.registry.get_glyph(glyph_type)
            
            # Create glyph for this operation
            glyph_id = f"op_{op_idx}"
            glyph = Glyph(
                id=glyph_id,
                symbol=glyph_info["symbol"],
                type=GlyphType.RECURSIVE,
                semantics=glyph_info["semantics"],
                position=(100 + op_depth * 80, 100 + op_idx * 30),  # Initial position
                size=10.0 - op_depth * 0.5,  # Size decreases with depth
                color=self._get_color_for_depth(op_depth),
                opacity=max(0.5, 1.0 - op_depth * 0.1),
                source_elements=[operation],
                description=f"Operation {op_type} at depth {op_depth}",
                metadata={
                    "operation_index": op_idx,
                    "operation_type": op_type,
                    "depth": op_depth,
                    "parameters": operation.get("parameters", {})
                }
            )
            glyph_map.glyphs.append(glyph)
            
            # Store glyph ID by depth
            depth_key = str(op_depth)
            if depth_key not in depth_glyphs:
                depth_glyphs[depth_key] = []
            if isinstance(depth_glyphs[depth_key], list):
                depth_glyphs[depth_key].append(glyph_id)
            else:
                depth_glyphs[depth_key] = [depth_glyphs[depth_key], glyph_id]
            
            # Connect to parent depth
            parent_depth_key = str(op_depth - 1)
            if parent_depth_key in depth_glyphs:
                parent_glyphs = depth_glyphs[parent_depth_key]
                if isinstance(parent_glyphs, list):
                    # Connect to closest parent in terms of operation index
                    parent_indices = [
                        int(g.split('_')[1]) if g.startswith('op_') else 0
                        for g in parent_glyphs
                    ]
                    closest_parent_idx = min(range(len(parent_indices)), key=lambda i: abs(parent_indices[i] - op_idx))
                    parent_glyph_id = parent_glyphs[closest_parent_idx]
                else:
                    parent_glyph_id = parent_glyphs
                
                glyph_map.connections.append(GlyphConnection(
                    source_id=parent_glyph_id,
                    target_id=glyph_id,
                    strength=0.8,
                    type="recursive_descent",
                    directed=True,
                    color=self._get_color_for_depth(op_depth - 1, alpha=0.6),
                    width=2.0 - op_depth * 0.2,
                    opacity=max(0.4, 0.9 - op_depth * 0.1)
                ))
        
        # Add result glyphs
        if trace_result:
            result_type = trace_result.get("type", "unknown")
            
            # Determine glyph type based on result type
            if "success" in result_type:
                glyph_type = "recursive_recursive_aegis"
            elif "collapse" in result_type:
                glyph_type = "meta_collapse_point"
            elif "partial" in result_type:
                glyph_type = "residue_ghost_activation"
            else:
                glyph_type = "meta_uncertainty"
            
            # Get glyph from registry
            glyph_info = self.registry.get_glyph(glyph_type)
            # Create result glyph
            result_glyph_id = "result"
            result_glyph = Glyph(
                id=result_glyph_id,
                symbol=glyph_info["symbol"],
                type=GlyphType.RECURSIVE if "success" in result_type else GlyphType.META,
                semantics=glyph_info["semantics"],
                position=(dimensions[0] / 2, dimensions[1] - 80),  # Bottom center
                size=15.0,
                color="#27ae60" if "success" in result_type else "#e74c3c",
                opacity=1.0,
                source_elements=[trace_result],
                description=f"Trace result: {result_type}",
                metadata={
                    "result_type": result_type,
                    "result_data": trace_result.get("data", {}),
                    "confidence": trace_result.get("confidence", 0.0)
                }
            )
            glyph_map.glyphs.append(result_glyph)
            
            # Connect deepest operations to result
            max_depth = max(int(d) for d in depth_glyphs.keys())
            deepest_glyphs = depth_glyphs[str(max_depth)]
            if isinstance(deepest_glyphs, list):
                for glyph_id in deepest_glyphs:
                    glyph_map.connections.append(GlyphConnection(
                        source_id=glyph_id,
                        target_id=result_glyph_id,
                        strength=0.9,
                        type="recursion_result",
                        directed=True,
                        color="#27ae60" if "success" in result_type else "#e74c3c",
                        width=1.5,
                        opacity=0.8
                    ))
            else:
                glyph_map.connections.append(GlyphConnection(
                    source_id=deepest_glyphs,
                    target_id=result_glyph_id,
                    strength=0.9,
                    type="recursion_result",
                    directed=True,
                    color="#27ae60" if "success" in result_type else "#e74c3c",
                    width=1.5,
                    opacity=0.8
                ))
            
            # Add seed to result connection
            glyph_map.connections.append(GlyphConnection(
                source_id="seed",
                target_id=result_glyph_id,
                strength=1.0,
                type="recursion_completion",
                directed=True,
                color="#9b59b6",  # Purple
                width=2.0,
                opacity=0.5
            ))
        
        # Add depth regions
        for depth in range(trace_depth + 1):
            depth_key = str(depth)
            if depth_key in depth_glyphs:
                depth_glyphs_list = depth_glyphs[depth_key]
                if not isinstance(depth_glyphs_list, list):
                    depth_glyphs_list = [depth_glyphs_list]
                glyph_map.regions[f"depth_{depth}"] = depth_glyphs_list
        
        # Apply layout if auto_layout is enabled
        if self.auto_layout:
            # For recursive traces, prefer hierarchical layout
            if layout_type == self.default_layout and self.default_layout != "hierarchical":
                layout_type = "hierarchical"
            glyph_map = self._apply_layout(glyph_map, layout_type)
        
        # Record execution time
        map_time = time.time() - map_start
        glyph_map.metadata["map_time"] = map_time
        
        # Add to history
        self.glyph_map_history.append(glyph_map)
        
        logger.info(f"Recursive trace mapping completed in {map_time:.2f}s")
        return glyph_map
    
    def combine_glyph_maps(
        self,
        glyph_maps: List[GlyphMap],
        layout_type: Optional[str] = None,
        dimensions: Optional[Tuple[int, int]] = None,
        scale: Optional[float] = None,
        connection_threshold: float = 0.5
    ) -> GlyphMap:
        """
        Combine multiple glyph maps into a unified map.
        
        Parameters:
        -----------
        glyph_maps : List[GlyphMap]
            Glyph maps to combine
        layout_type : Optional[str]
            Type of layout to use
        dimensions : Optional[Tuple[int, int]]
            Dimensions for visualization
        scale : Optional[float]
            Scale factor for visualization
        connection_threshold : float
            Minimum strength for inter-map connections
            
        Returns:
        --------
        GlyphMap
            Combined glyph map
        """
        map_start = time.time()
        layout_type = layout_type or self.default_layout
        dimensions = dimensions or (
            max(gm.dimensions[0] for gm in glyph_maps),
            max(gm.dimensions[1] for gm in glyph_maps)
        )
        scale = scale or self.default_scale
        
        logger.info(f"Combining {len(glyph_maps)} glyph maps")
        
        # Create unique ID for combined glyph map
        map_id = f"combined_{int(time.time())}_{hashlib.md5(str([gm.id for gm in glyph_maps]).encode()).hexdigest()[:8]}"
        
        # Initialize combined glyph map
        combined_map = GlyphMap(
            id=map_id,
            glyphs=[],
            connections=[],
            source_type="combined",
            layout_type=layout_type,
            dimensions=dimensions,
            scale=scale,
            metadata={
                "source_maps": [gm.id for gm in glyph_maps],
                "source_types": [gm.source_type for gm in glyph_maps],
                "timestamp": time.time()
            }
        )
        
        # Generate prefix mappings to ensure unique IDs
        id_mapping = {}
        for i, gm in enumerate(glyph_maps):
            prefix = f"map{i}_"
            for glyph in gm.glyphs:
                id_mapping[glyph.id] = prefix + glyph.id
        
        # Add glyphs from each map
        for i, gm in enumerate(glyph_maps):
            prefix = f"map{i}_"
            
            # Add region for this map
            map_region = f"map_{i}"
            combined_map.regions[map_region] = []
            
            # Add glyphs with prefixed IDs
            for glyph in gm.glyphs:
                new_id = prefix + glyph.id
                new_glyph = Glyph(
                    id=new_id,
                    symbol=glyph.symbol,
                    type=glyph.type,
                    semantics=glyph.semantics,
                    position=glyph.position,  # Will be updated by layout
                    size=glyph.size,
                    color=glyph.color,
                    opacity=glyph.opacity,
                    source_elements=glyph.source_elements,
                    description=glyph.description,
                    metadata={
                        **glyph.metadata,
                        "original_id": glyph.id,
                        "source_map": gm.id
                    }
                )
                combined_map.glyphs.append(new_glyph)
                combined_map.regions[map_region].append(new_id)
            
            # Add connections with prefixed IDs
            for conn in gm.connections:
                combined_map.connections.append(GlyphConnection(
                    source_id=prefix + conn.source_id,
                    target_id=prefix + conn.target_id,
                    strength=conn.strength,
                    type=conn.type,
                    directed=conn.directed,
                    color=conn.color,
                    width=conn.width,
                    opacity=conn.opacity,
                    metadata={
                        **conn.metadata,
                        "source_map": gm.id
                    }
                ))
            
            # Add focal points with prefixed IDs
            for focal_point in gm.focal_points:
                combined_map.focal_points.append(prefix + focal_point)
        
        # Create connections between maps for related glyphs
        for i, gm1 in enumerate(glyph_maps):
            for j, gm2 in enumerate(glyph_maps):
                if i >= j:
                    continue  # Skip self-connections and duplicates
                
                # Find related glyphs between maps
                related_pairs = self._find_related_glyphs(gm1, gm2)
                
                # Add connections for sufficiently related glyphs
                for glyph1_id, glyph2_id, similarity in related_pairs:
                    if similarity >= connection_threshold:
                        prefixed_id1 = f"map{i}_{glyph1_id}"
                        prefixed_id2 = f"map{j}_{glyph2_id}"
                        
                        combined_map.connections.append(GlyphConnection(
                            source_id=prefixed_id1,
                            target_id=prefixed_id2,
                            strength=similarity,
                            type="cross_map_relation",
                            directed=False,
                            color="#3498db",  # Blue
                            width=1.0 + similarity,
                            opacity=0.6 * similarity,
                            metadata={
                                "relation_type": "cross_map",
                                "source_map": gm1.id,
                                "target_map": gm2.id,
                                "similarity": similarity
                            }
                        ))
        
        # Apply layout if auto_layout is enabled
        if self.auto_layout:
            combined_map = self._apply_layout(combined_map, layout_type)
        
        # Record execution time
        map_time = time.time() - map_start
        combined_map.metadata["map_time"] = map_time
        
        # Add to history
        self.glyph_map_history.append(combined_map)
        
        logger.info(f"Glyph map combination completed in {map_time:.2f}s")
        return combined_map
    
    def visualize(
        self,
        glyph_map: GlyphMap,
        output_path: Optional[str] = None,
        interactive: bool = True
    ) -> Any:
        """
        Visualize a glyph map.
        
        Parameters:
        -----------
        glyph_map : GlyphMap
            Glyph map to visualize
        output_path : Optional[str]
            Path to save visualization to
        interactive : bool
            Whether to generate interactive visualization
            
        Returns:
        --------
        Any
            Visualization result
        """
        if self.visualizer:
            return self.visualizer.visualize_glyph_map(
                glyph_map=glyph_map,
                output_path=output_path,
                interactive=interactive
            )
        else:
            # Simple matplotlib visualization if no visualizer
            return self._simple_visualization(
                glyph_map=glyph_map,
                output_path=output_path
            )
    
    def save_glyph_map(
        self,
        glyph_map: GlyphMap,
        output_path: str
    ) -> str:
        """
        Save a glyph map to a file.
        
        Parameters:
        -----------
        glyph_map : GlyphMap
            Glyph map to save
        output_path : str
            Path to save glyph map to
            
        Returns:
        --------
        str
            Path to saved file
        """
        # Convert to serializable format
        serializable_map = {
            "id": glyph_map.id,
            "source_type": glyph_map.source_type,
            "layout_type": glyph_map.layout_type,
            "dimensions": glyph_map.dimensions,
            "scale": glyph_map.scale,
            "focal_points": glyph_map.focal_points,
            "regions": glyph_map.regions,
            "metadata": glyph_map.metadata,
            "glyphs": [
                {
                    "id": g.id,
                    "symbol": g.symbol,
                    "type": g.type.value,
                    "semantics": [s.value for s in g.semantics],
                    "position": g.position,
                    "size": g.size,
                    "color": g.color,
                    "opacity": g.opacity,
                    "description": g.description,
                    "metadata": g.metadata
                }
                for g in glyph_map.glyphs
            ],
            "connections": [
                {
                    "source_id": c.source_id,
                    "target_id": c.target_id,
                    "strength": c.strength,
                    "type": c.type,
                    "directed": c.directed,
                    "color": c.color,
                    "width": c.width,
                    "opacity": c.opacity,
                    "metadata": c.metadata
                }
                for c in glyph_map.connections
            ]
        }
        
        # Ensure directory exists
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save to file
        with open(output_path, "w") as f:
            json.dump(serializable_map, f, indent=2)
        
        logger.info(f"Saved glyph map to {output_path}")
        return output_path
    
    def load_glyph_map(
        self,
        input_path: str
    ) -> GlyphMap:
        """
        Load a glyph map from a file.
        
        Parameters:
        -----------
        input_path : str
            Path to load glyph map from
            
        Returns:
        --------
        GlyphMap
            Loaded glyph map
        """
        # Load from file
        with open(input_path, "r") as f:
            data = json.load(f)
        
        # Convert to GlyphMap
        glyphs = [
            Glyph(
                id=g["id"],
                symbol=g["symbol"],
                type=GlyphType(g["type"]),
                semantics=[GlyphSemantic(s) for s in g["semantics"]],
                position=tuple(g["position"]),
                size=g["size"],
                color=g["color"],
                opacity=g["opacity"],
                description=g.get("description"),
                metadata=g.get("metadata", {})
            )
            for g in data["glyphs"]
        ]
        
        connections = [
            GlyphConnection(
                source_id=c["source_id"],
                target_id=c["target_id"],
                strength=c["strength"],
                type=c["type"],
                directed=c["directed"],
                color=c["color"],
                width=c["width"],
                opacity=c["opacity"],
                metadata=c.get("metadata", {})
            )
            for c in data["connections"]
        ]
        
        glyph_map = GlyphMap(
            id=data["id"],
            glyphs=glyphs,
            connections=connections,
            source_type=data["source_type"],
            layout_type=data["layout_type"],
            dimensions=tuple(data["dimensions"]),
            scale=data["scale"],
            focal_points=data.get("focal_points", []),
            regions=data.get("regions", {}),
            metadata=data.get("metadata", {})
        )
        
        logger.info(f"Loaded glyph map from {input_path}")
        return glyph_map
    
    # Helper methods
    
    def _apply_layout(
        self,
        glyph_map: GlyphMap,
        layout_type: str
    ) -> GlyphMap:
        """Apply a layout to a glyph map."""
        layout_start = time.time()
        
        if layout_type == "force_directed":
            glyph_map = self._apply_force_directed_layout(glyph_map)
        elif layout_type == "hierarchical":
            glyph_map = self._apply_hierarchical_layout(glyph_map)
        elif layout_type == "circular":
            glyph_map = self._apply_circular_layout(glyph_map)
        elif layout_type == "grid":
            glyph_map = self._apply_grid_layout(glyph_map)
        elif layout_type == "radial":
            glyph_map = self._apply_radial_layout(glyph_map)
        else:
            logger.warning(f"Unknown layout type: {layout_type}, using force_directed")
            glyph_map = self._apply_force_directed_layout(glyph_map)
        
        layout_time = time.time() - layout_start
        logger.info(f"Applied {layout_type} layout in {layout_time:.2f}s")
        
        return glyph_map
    
    def _apply_force_directed_layout(self, glyph_map: GlyphMap) -> GlyphMap:
        """Apply force-directed layout to a glyph map."""
        # Create networkx graph
        G = nx.Graph()
        
        # Add nodes
        for glyph in glyph_map.glyphs:
            G.add_node(glyph.id, size=glyph.size, type=glyph.type.value)
        
        # Add edges
        for conn in glyph_map.connections:
            if conn.source_id in G and conn.target_id in G:
                G.add_edge(
                    conn.source_id,
                    conn.target_id,
                    weight=conn.strength
                )
        
        # Apply force-directed layout
        width, height = glyph_map.dimensions
        pos = nx.spring_layout(
            G,
            k=0.2,  # Optimal distance between nodes
            iterations=100,
            seed=42
        )
        
        # Scale and shift positions to fit dimensions
        pos_array = np.array(list(pos.values()))
        if len(pos_array) > 0:
            min_x, min_y = pos_array.min(axis=0)
            max_x, max_y = pos_array.max(axis=0)
            
            # Avoid division by zero
            x_range = max_x - min_x
            y_range = max_y - min_y
            
            if x_range > 0:
                scale_x = (width * 0.8) / x_range
            else:
                scale_x = 1.0
                
            if y_range > 0:
                scale_y = (height * 0.8) / y_range
            else:
                scale_y = 1.0
            
            # Apply scaling
            for node_id, (x, y) in pos.items():
                x_scaled = ((x - min_x) * scale_x) + width * 0.1
                y_scaled = ((y - min_y) * scale_y) + height * 0.1
                pos[node_id] = (x_scaled, y_scaled)
        
        # Update glyph positions
        for glyph in glyph_map.glyphs:
            if glyph.id in pos:
                glyph.position = pos[glyph.id]
        
        return glyph_map
    
    def _apply_hierarchical_layout(self, glyph_map: GlyphMap) -> GlyphMap:
        """Apply hierarchical layout to a glyph map."""
        # Create directed graph
        G = nx.DiGraph()
        
        # Add nodes
        for glyph in glyph_map.glyphs:
            G.add_node(glyph.id, size=glyph.size, type=glyph.type.value)
        
        # Add directed edges
        for conn in glyph_map.connections:
            if conn.directed and conn.source_id in G and conn.target_id in G:
                G.add_edge(
                    conn.source_id,
                    conn.target_id,
                    weight=conn.strength
                )
        
        # Find root nodes (nodes with no incoming edges)
        root_nodes = [n for n in G.nodes() if G.in_degree(n) == 0]
        
        # If no root nodes, use focal points or any node
        if not root_nodes:
            if glyph_map.focal_points:
                root_nodes = [fp for fp in glyph_map.focal_points if fp in G]
            else:
                root_nodes = [glyph_map.glyphs[0].id] if glyph_map.glyphs else []
        
        # Apply hierarchical layout
        width, height = glyph_map.dimensions
        
        # If we have root nodes, use them
        if root_nodes:
            # Create layers
            layers = {}
            visited = set()
            
            # BFS to assign layers
            current_layer = root_nodes
            layer_idx = 0
            
            while current_layer and layer_idx < 20:  # Limit to 20 layers to prevent infinite loops
                layers[layer_idx] = current_layer
                next_layer = []
                
                for node in current_layer:
                    visited.add(node)
                    for _, neighbor in G.out_edges(node):
                        if neighbor not in visited and neighbor not in next_layer:
                            next_layer.append(neighbor)
                
                current_layer = next_layer
                layer_idx += 1
            
            # Place nodes by layer
            for layer_idx, nodes in layers.items():
                y_pos = (layer_idx + 1) * (height / (len(layers) + 1))
                x_step = width / (len(nodes) + 1)
                
                for i, node_id in enumerate(nodes):
                    x_pos = (i + 1) * x_step
                    # Find and update glyph position
                    for glyph in glyph_map.glyphs:
                        if glyph.id == node_id:
                            glyph.position = (x_pos, y_pos)
                            break
            
            # Assign positions to any unvisited nodes
            unvisited = [g.id for g in glyph_map.glyphs if g.id not in visited]
            if unvisited:
                y_pos = (layer_idx + 1) * (height / (len(layers) + 2))
                x_step = width / (len(unvisited) + 1)
                
                for i, node_id in enumerate(unvisited):
                    x_pos = (i + 1) * x_step
                    # Find and update glyph position
                    for glyph in glyph_map.glyphs:
                        if glyph.id == node_id:
                            glyph.position = (x_pos, y_pos)
                            break
        else:
            # Fallback to simple grid layout
            glyph_map = self._apply_grid_layout(glyph_map)
        
        return glyph_map
    
    def _apply_circular_layout(self, glyph_map: GlyphMap) -> GlyphMap:
        """Apply circular layout to a glyph map."""
        # Create networkx graph
        G = nx.Graph()
        
        # Add nodes
        for glyph in glyph_map.glyphs:
            G.add_node(glyph.id, size=glyph.size, type=glyph.type.value)
        
        # Add edges
        for conn in glyph_map.connections:
            if conn.source_id in G and conn.target_id in G:
                G.add_edge(
                    conn.source_id,
                    conn.target_id,
                    weight=conn.strength
                )
        
        # Apply circular layout
        width, height = glyph_map.dimensions
        center_x, center_y = width / 2, height / 2
        radius = min(width, height) * 0.4
        
        pos = nx.circular_layout(G, scale=radius)
        
        # Center the layout
        for node_id, (x, y) in pos.items():
            pos[node_id] = (x + center_x, y + center_y)
        
        # Update glyph positions
        for glyph in glyph_map.glyphs:
            if glyph.id in pos:
                glyph.position = pos[glyph.id]
        
        return glyph_map
    
    def _apply_grid_layout(self, glyph_map: GlyphMap) -> GlyphMap:
        """Apply grid layout to a glyph map."""
        width, height = glyph_map.dimensions
        num_glyphs = len(glyph_map.glyphs)
        
        # Calculate grid dimensions
        grid_size = int(np.ceil(np.sqrt(num_glyphs)))
        cell_width = width / (grid_size + 1)
        cell_height = height / (grid_size + 1)
        
        # Assign positions
        for i, glyph in enumerate(glyph_map.glyphs):
            row = i // grid_size
            col = i % grid_size
            
            glyph.position = (
                (col + 1) * cell_width,
                (row + 1) * cell_height
            )
        
        return glyph_map
    
    def _apply_radial_layout(self, glyph_map: GlyphMap) -> GlyphMap:
        """Apply radial layout to a glyph map."""
        # Create networkx graph
        G = nx.Graph()
        
        # Add nodes
        for glyph in glyph_map.glyphs:
            G.add_node(glyph.id, size=glyph.size, type=glyph.type.value)
        
        # Add edges
        for conn in glyph_map.connections:
            if conn.source_id in G and conn.target_id in G:
                G.add_edge(
                    conn.source_id,
                    conn.target_id,
                    weight=conn.strength
                )
        
        # Determine central nodes
        if glyph_map.focal_points:
            central_nodes = [fp for fp in glyph_map.focal_points if fp in G]
        else:
            # Use betweenness centrality to find central nodes
            centrality = nx.betweenness_centrality(G)
            central_nodes = sorted(
                centrality.keys(),
                key=lambda x: centrality[x],
                reverse=True
            )[:min(3, len(centrality))]
        
        # Apply radial layout
        width, height = glyph_map.dimensions
        center_x, center_y = width / 2, height / 2
        
        pos = nx.kamada_kawai_layout(G)
        
        # Scale and shift positions to fit dimensions
        pos_array = np.array(list(pos.values()))
        if len(pos_array) > 0:
            min_x, min_y = pos_array.min(axis=0)
            max_x, max_y = pos_array.max(axis=0)
            
            # Avoid division by zero
            x_range = max_x - min_x
            y_range = max_y - min_y
            
            if x_range > 0:
                scale_x = (width * 0.8) / x_range
            else:
                scale_x = 1.0
                
            if y_range > 0:
                scale_y = (height * 0.8) / y_range
            else:
                scale_y = 1.0
            
            # Apply scaling
            for node_id, (x, y) in pos.items():
                x_scaled = ((x - min_x) * scale_x) + width * 0.1
                y_scaled = ((y - min_y) * scale_y) + height * 0.1
                pos[node_id] = (x_scaled, y_scaled)
        
        # Force central nodes to center
        if central_nodes:
            # Place central nodes near center
            angle_step = 2 * np.pi / len(central_nodes)
            center_radius = min(width, height) * 0.15
            
            for i, node_id in enumerate(central_nodes):
                angle = i * angle_step
                x = center_x + center_radius * np.cos(angle)
                y = center_y + center_radius * np.sin(angle)
                pos[node_id] = (x, y)
        
        # Update glyph positions
        for glyph in glyph_map.glyphs:
            if glyph.id in pos:
                glyph.position = pos[glyph.id]
        
        return glyph_map
    
    def _apply_focus(
        self,
        glyph_map: GlyphMap,
        focus_on: List[str]
    ) -> GlyphMap:
        """Apply focus to specific tokens or elements."""
        # Find token glyphs matching focus terms
        focus_glyph_ids = []
        
        for glyph in glyph_map.glyphs:
            # Check if glyph contains any focus term
            if glyph.type == GlyphType.SENTINEL and hasattr(glyph, 'source_elements') and glyph.source_elements:
                for term in focus_on:
                    if any(term in str(elem) for elem in glyph.source_elements):
                        focus_glyph_ids.append(glyph.id)
                        break
        
        if not focus_glyph_ids:
            # No matching token glyphs found
            return glyph_map
        
        # Update focal points
        glyph_map.focal_points = focus_glyph_ids
        
        # Increase size and opacity of focal glyphs
        for glyph in glyph_map.glyphs:
            if glyph.id in focus_glyph_ids:
                glyph.size *= 1.5
                glyph.opacity = min(1.0, glyph.opacity + 0.2)
        
        # Highlight connections to focal glyphs
        for conn in glyph_map.connections:
            if conn.source_id in focus_glyph_ids or conn.target_id in focus_glyph_ids:
                conn.width *= 1.5
                conn.opacity = min(1.0, conn.opacity + 0.2)
        
        return glyph_map
    
    def _find_related_glyphs(
        self,
        glyph_map1: GlyphMap,
        glyph_map2: GlyphMap
    ) -> List[Tuple[str, str, float]]:
        """Find related glyphs between two glyph maps."""
        related_pairs = []
        
        # Define similarity functions based on glyph type
        def token_similarity(g1: Glyph, g2: Glyph) -> float:
            """Calculate similarity between token glyphs."""
            if (g1.type == GlyphType.SENTINEL and g2.type == GlyphType.SENTINEL and
                hasattr(g1, 'source_elements') and hasattr(g2, 'source_elements') and
                g1.source_elements and g2.source_elements):
                # Compare token text
                text1 = str(g1.source_elements[0])
                text2 = str(g2.source_elements[0])
                if text1 == text2:
                    return 1.0
                elif text1 in text2 or text2 in text1:
                    return 0.8
                else:
                    # Compute string similarity
                    return 1.0 - min(1.0, distance.levenshtein(text1, text2) / max(len(text1), len(text2)))
            return 0.0
        
        def attribution_similarity(g1: Glyph, g2: Glyph) -> float:
            """Calculate similarity between attribution glyphs."""
            if g1.type == GlyphType.ATTRIBUTION and g2.type == GlyphType.ATTRIBUTION:
                # Compare attribution metadata
                metadata_sim = 0.0
                count = 0
                
                # Compare attributes if they exist in both
                for attr in ['source_index', 'target_index', 'attribution_type', 'strength']:
                    if attr in g1.metadata and attr in g2.metadata:
                        if g1.metadata[attr] == g2.metadata[attr]:
                            metadata_sim += 1.0
                        else:
                            # For numeric values, calculate relative similarity
                            if attr == 'strength' and isinstance(g1.metadata[attr], (int, float)) and isinstance(g2.metadata[attr], (int, float)):
                                diff = abs(g1.metadata[attr] - g2.metadata[attr])
                                metadata_sim += max(0.0, 1.0 - diff)
                            else:
                                metadata_sim += 0.0
                        count += 1
                
                # Symbol similarity
                symbol_sim = 1.0 if g1.symbol == g2.symbol else 0.0
                
                # Combine similarities
                if count > 0:
                    return (metadata_sim / count) * 0.7 + symbol_sim * 0.3
                else:
                    return symbol_sim
            return 0.0
        
        def residue_similarity(g1: Glyph, g2: Glyph) -> float:
            """Calculate similarity between residue glyphs."""
            if g1.type == GlyphType.RESIDUE and g2.type == GlyphType.RESIDUE:
                # Check if they represent the same residue type
                if g1.metadata.get('pattern_type') == g2.metadata.get('pattern_type'):
                    return 0.9
                
                # Compare symbols
                if g1.symbol == g2.symbol:
                    return 0.7
                
                # Compare confidence if available
                if 'confidence' in g1.metadata and 'confidence' in g2.metadata:
                    conf_diff = abs(g1.metadata['confidence'] - g2.metadata['confidence'])
                    return max(0.0, 0.5 - conf_diff)
                
                return 0.3  # Different residue types but still residues
            return 0.0
        
        def recursive_similarity(g1: Glyph, g2: Glyph) -> float:
            """Calculate similarity between recursive glyphs."""
            if g1.type == GlyphType.RECURSIVE and g2.type == GlyphType.RECURSIVE:
                # Compare symbols
                if g1.symbol == g2.symbol:
                    return 0.8
                
                # Compare depth if available
                if 'depth' in g1.metadata and 'depth' in g2.metadata:
                    if g1.metadata['depth'] == g2.metadata['depth']:
                        return 0.7
                    else:
                        depth_diff = abs(g1.metadata['depth'] - g2.metadata['depth'])
                        return max(0.0, 0.6 - (depth_diff * 0.1))
                
                return 0.4  # Different recursive types but still recursive
            return 0.0
        
        def meta_similarity(g1: Glyph, g2: Glyph) -> float:
            """Calculate similarity between meta glyphs."""
            if g1.type == GlyphType.META and g2.type == GlyphType.META:
                # Compare symbols
                if g1.symbol == g2.symbol:
                    return 0.9
                
                # Compare semantics
                common_semantics = set(s.value for s in g1.semantics).intersection(
                    set(s.value for s in g2.semantics)
                )
                if common_semantics:
                    return 0.6 + 0.3 * (len(common_semantics) / max(len(g1.semantics), len(g2.semantics)))
                
                return 0.3  # Different meta types but still meta
            return 0.0
        
        # Apply appropriate similarity function based on glyph types
        for g1 in glyph_map1.glyphs:
            for g2 in glyph_map2.glyphs:
                similarity = 0.0
                
                # Apply type-specific similarity function
                if g1.type == GlyphType.SENTINEL and g2.type == GlyphType.SENTINEL:
                    similarity = token_similarity(g1, g2)
                elif g1.type == GlyphType.ATTRIBUTION and g2.type == GlyphType.ATTRIBUTION:
                    similarity = attribution_similarity(g1, g2)
                elif g1.type == GlyphType.RESIDUE and g2.type == GlyphType.RESIDUE:
                    similarity = residue_similarity(g1, g2)
                elif g1.type == GlyphType.RECURSIVE and g2.type == GlyphType.RECURSIVE:
                    similarity = recursive_similarity(g1, g2)
                elif g1.type == GlyphType.META and g2.type == GlyphType.META:
                    similarity = meta_similarity(g1, g2)
                elif g1.type == g2.type:
                    # Same type but not handled above
                    if g1.symbol == g2.symbol:
                        similarity = 0.6
                    else:
                        similarity = 0.3
                
                # Add if similarity is significant
                if similarity >= 0.5:
                    related_pairs.append((g1.id, g2.id, similarity))
        
        # Sort by similarity (highest first)
        related_pairs.sort(key=lambda x: x[2], reverse=True)
        
        return related_pairs
    
    def _calculate_signature_similarity(
        self, 
        signature1: str, 
        signature2: str
    ) -> float:
        """Calculate similarity between two residue signatures."""
        # Normalize signatures
        sig1 = signature1.lower()
        sig2 = signature2.lower()
        
        # Calculate Levenshtein distance
        max_len = max(len(sig1), len(sig2))
        if max_len == 0:
            return 1.0  # Both empty
            
        lev_dist = distance.levenshtein(sig1, sig2)
        sim = 1.0 - (lev_dist / max_len)
        
        # Boost similarity for common prefixes
        common_prefix_len = 0
        for i in range(min(len(sig1), len(sig2))):
            if sig1[i] == sig2[i]:
                common_prefix_len += 1
            else:
                break
        
        prefix_boost = 0.0
        if common_prefix_len > 3:  # At least a few characters
            prefix_boost = min(0.2, common_prefix_len / max_len)
        
        return min(1.0, sim + prefix_boost)
    
    def _calculate_context_similarity(
        self, 
        context1: Dict[str, Any], 
        context2: Dict[str, Any]
    ) -> float:
        """Calculate similarity between two residue contexts."""
        # Handle empty contexts
        if not context1 or not context2:
            return 0.0
        
        # Find common keys
        common_keys = set(context1.keys()).intersection(set(context2.keys()))
        if not common_keys:
            return 0.0
        
        # Calculate similarity for each common key
        similarity_sum = 0.0
        for key in common_keys:
            val1 = context1[key]
            val2 = context2[key]
            
            if isinstance(val1, str) and isinstance(val2, str):
                # String similarity
                similarity_sum += self._calculate_signature_similarity(val1, val2)
            elif isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                # Numeric similarity
                max_val = max(abs(val1), abs(val2))
                if max_val > 0:
                    similarity_sum += 1.0 - min(1.0, abs(val1 - val2) / max_val)
                else:
                    similarity_sum += 1.0  # Both zero
            elif val1 == val2:
                # Other types, check equality
                similarity_sum += 1.0
            else:
                similarity_sum += 0.0
        
        # Average similarity
        return similarity_sum / len(common_keys)
    
    def _get_color_for_attribution(self, link: AttributionLink) -> str:
        """Get color for attribution link based on type and strength."""
        if link.attribution_type == AttributionType.DIRECT:
            # Blues for direct attribution, darker with strength
            blue_val = max(0, int(255 - (link.strength * 170)))
            return f"rgb(0, {120 + int(link.strength * 70)}, {180 + blue_val})"
        elif link.attribution_type == AttributionType.INDIRECT:
            # Purples for indirect attribution
            return f"rgb({120 + int(link.strength * 70)}, 0, {180 + int(link.strength * 60)})"
        elif link.attribution_type == AttributionType.RESIDUAL:
            # Greens for residual attribution
            return f"rgb(0, {150 + int(link.strength * 70)}, {80 + int(link.strength * 40)})"
        elif link.attribution_type == AttributionType.RECURSIVE:
            # Orange-reds for recursive attribution
            return f"rgb({200 + int(link.strength * 50)}, {70 + int(link.strength * 60)}, 0)"
        else:
            # Default gray
            intensity = 100 + int(link.strength * 100)
            return f"rgb({intensity}, {intensity}, {intensity})"
    
    def _get_color_for_residue_type(self, residue_type: str) -> str:
        """Get color for residue based on type."""
        colors = {
            "memory_decay": "#3498db",       # Blue
            "value_conflict": "#e74c3c",     # Red
            "ghost_activation": "#9b59b6",   # Purple
            "boundary_hesitation": "#f39c12", # Orange
            "null_output": "#95a5a6",        # Gray
            "recursive_collapse": "#27ae60", # Green
            "attention_drift": "#1abc9c",    # Turquoise
            "token_oscillation": "#d35400"   # Dark Orange
        }
        
        return colors.get(residue_type, "#2c3e50")  # Default dark blue
    
    def _get_color_for_layer(self, layer: int, alpha: float = 1.0) -> str:
        """Get color for a specific layer."""
        # HSL color with hue based on layer
        hue = (layer * 30) % 360
        return f"hsl({hue}, 70%, 60%, {alpha})"
    
    def _get_color_for_depth(self, depth: int, alpha: float = 1.0) -> str:
        """Get color for a specific recursion depth."""
        # Base color shifts from purple to blue to green with increasing depth
        if depth == 0:
            return f"rgba(155, 89, 182, {alpha})"  # Purple
        elif depth == 1:
            return f"rgba(52, 152, 219, {alpha})"  # Blue
        elif depth == 2:
            return f"rgba(26, 188, 156, {alpha})"  # Turquoise
        elif depth == 3:
            return f"rgba(39, 174, 96, {alpha})"   # Green
        elif depth == 4:
            return f"rgba(241, 196, 15, {alpha})"  # Yellow
        elif depth >= 5:
            return f"rgba(230, 126, 34, {alpha})"  # Orange
    
    def _scale_positions(
        self,
        positions: np.ndarray,
        dimensions: Tuple[int, int]
    ) -> np.ndarray:
        """Scale positions to fit dimensions."""
        width, height = dimensions
        
        # Get bounds
        min_x, min_y = positions.min(axis=0)
        max_x, max_y = positions.max(axis=0)
        
        # Avoid division by zero
        x_range = max_x - min_x
        y_range = max_y - min_y
        
        if x_range > 0:
            scale_x = (width * 0.8) / x_range
        else:
            scale_x = 1.0
            
        if y_range > 0:
            scale_y = (height * 0.8) / y_range
        else:
            scale_y = 1.0
        
        # Apply scaling and shift
        positions_scaled = np.zeros_like(positions)
        positions_scaled[:, 0] = (positions[:, 0] - min_x) * scale_x + width * 0.1
        positions_scaled[:, 1] = (positions[:, 1] - min_y) * scale_y + height * 0.1
        
        return positions_scaled
    
    def _simple_visualization(
        self,
        glyph_map: GlyphMap,
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Simple matplotlib visualization if no visualizer available."""
        # Create figure
        plt.figure(figsize=(12, 10))
        
        # Plot connections
        for conn in glyph_map.connections:
            # Find source and target glyphs
            source_glyph = next((g for g in glyph_map.glyphs if g.id == conn.source_id), None)
            target_glyph = next((g for g in glyph_map.glyphs if g.id == conn.target_id), None)
            
            if source_glyph and target_glyph:
                # Get positions
                source_x, source_y = source_glyph.position
                target_x, target_y = target_glyph.position
                
                # Draw connection
                plt.plot(
                    [source_x, target_x],
                    [source_y, target_y],
                    color=conn.color,
                    linewidth=conn.width,
                    alpha=conn.opacity,
                    zorder=1,
                    linestyle='-' if conn.directed else '--'
                )
                
                # Add arrow if directed
                if conn.directed:
                    dx = target_x - source_x
                    dy = target_y - source_y
                    dist = np.sqrt(dx**2 + dy**2)
                    if dist > 0:
                        # Normalize and scale
                        dx, dy = dx / dist, dy / dist
                        midpoint_x = (source_x + target_x) / 2
                        midpoint_y = (source_y + target_y) / 2
                        
                        # Draw arrowhead
                        plt.arrow(
                            midpoint_x - dx * 5,
                            midpoint_y - dy * 5,
                            dx * 10,
                            dy * 10,
                            head_width=5,
                            head_length=5,
                            fc=conn.color,
                            ec=conn.color,
                            alpha=conn.opacity,
                            zorder=1
                        )
        
        # Plot glyphs
        for glyph in glyph_map.glyphs:
            x, y = glyph.position
            
            # Draw glyph as text
            plt.text(
                x, y,
                glyph.symbol,
                fontsize=glyph.size,
                color=glyph.color,
                alpha=glyph.opacity,
                ha='center',
                va='center',
                zorder=2
            )
            
            # Draw circle around focal points
            if glyph.id in glyph_map.focal_points:
                circle = plt.Circle(
                    (x, y),
                    glyph.size * 0.8,
                    fill=False,
                    color='black',
                    linestyle=':',
                    alpha=0.7,
                    zorder=1
                )
                plt.gca().add_patch(circle)
        
        # Set plot limits
        width, height = glyph_map.dimensions
        plt.xlim(0, width)
        plt.ylim(0, height)
        
        # Remove axes
        plt.axis('off')
        
        # Add title
        title = f"Glyph Map: {glyph_map.source_type.capitalize()}"
        if "trace_target" in glyph_map.metadata:
            title += f" - {glyph_map.metadata['trace_target']}"
        plt.title(title)
        
        # Save if output path provided
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            return {"output_path": output_path}
        
        # Return figure data
        return {"figure": plt.gcf()}


# Helper class for glyph map exploration
class GlyphExplorer:
    """
    Utility class for interactive exploration of glyph maps.
    
    This class provides methods for filtering, searching, and analyzing
    glyph maps to extract insights and patterns.
    """
    
    def __init__(self, glyph_map: GlyphMap):
        """
        Initialize the glyph explorer.
        
        Parameters:
        -----------
        glyph_map : GlyphMap
            Glyph map to explore
        """
        self.glyph_map = glyph_map
        self.filtered_glyphs = glyph_map.glyphs
        self.filtered_connections = glyph_map.connections
    
    def filter_by_type(self, glyph_type: GlyphType) -> 'GlyphExplorer':
        """
        Filter glyphs by type.
        
        Parameters:
        -----------
        glyph_type : GlyphType
            Type of glyphs to include
            
        Returns:
        --------
        GlyphExplorer
            Self, for method chaining
        """
        self.filtered_glyphs = [
            g for g in self.filtered_glyphs
            if g.type == glyph_type
        ]
        self._update_connections()
        return self
    
    def filter_by_semantic(self, semantic: GlyphSemantic) -> 'GlyphExplorer':
        """
        Filter glyphs by semantic dimension.
        
        Parameters:
        -----------
        semantic : GlyphSemantic
            Semantic dimension to filter by
            
        Returns:
        --------
        GlyphExplorer
            Self, for method chaining
        """
        self.filtered_glyphs = [
            g for g in self.filtered_glyphs
            if semantic in g.semantics
        ]
        self._update_connections()
        return self
    
    def filter_by_symbol(self, symbol: str) -> 'GlyphExplorer':
        """
        Filter glyphs by symbol.
        
        Parameters:
        -----------
        symbol : str
            Symbol to filter by
            
        Returns:
        --------
        GlyphExplorer
            Self, for method chaining
        """
        self.filtered_glyphs = [
            g for g in self.filtered_glyphs
            if g.symbol == symbol
        ]
        self._update_connections()
        return self
    
    def filter_by_size(
        self,
        min_size: Optional[float] = None,
        max_size: Optional[float] = None
    ) -> 'GlyphExplorer':
        """
        Filter glyphs by size.
        
        Parameters:
        -----------
        min_size : Optional[float]
            Minimum size (inclusive)
        max_size : Optional[float]
            Maximum size (inclusive)
            
        Returns:
        --------
        GlyphExplorer
            Self, for method chaining
        """
        if min_size is not None:
            self.filtered_glyphs = [
                g for g in self.filtered_glyphs
                if g.size >= min_size
            ]
        
        if max_size is not None:
            self.filtered_glyphs = [
                g for g in self.filtered_glyphs
                if g.size <= max_size
            ]
        
        self._update_connections()
        return self
    
    def filter_by_metadata(
        self,
        key: str,
        value: Any
    ) -> 'GlyphExplorer':
        """
        Filter glyphs by metadata field.
        
        Parameters:
        -----------
        key : str
            Metadata key
        value : Any
            Metadata value to match
            
        Returns:
        --------
        GlyphExplorer
            Self, for method chaining
        """
        self.filtered_glyphs = [
            g for g in self.filtered_glyphs
            if key in g.metadata and g.metadata[key] == value
        ]
        self._update_connections()
        return self
    
    def filter_connections_by_type(self, conn_type: str) -> 'GlyphExplorer':
        """
        Filter connections by type.
        
        Parameters:
        -----------
        conn_type : str
            Connection type to filter by
            
        Returns:
        --------
        GlyphExplorer
            Self, for method chaining
        """
        self.filtered_connections = [
            c for c in self.filtered_connections
            if c.type == conn_type
        ]
        return self
    
    def filter_connections_by_strength(
        self,
        min_strength: Optional[float] = None,
        max_strength: Optional[float] = None
    ) -> 'GlyphExplorer':
        """
        Filter connections by strength.
        
        Parameters:
        -----------
        min_strength : Optional[float]
            Minimum strength (inclusive)
        max_strength : Optional[float]
            Maximum strength (inclusive)
            
        Returns:
        --------
        GlyphExplorer
            Self, for method chaining
        """
        if min_strength is not None:
            self.filtered_connections = [
                c for c in self.filtered_connections
                if c.strength >= min_strength
            ]
        
        if max_strength is not None:
            self.filtered_connections = [
                c for c in self.filtered_connections
                if c.strength <= max_strength
            ]
        
        return self
    
    def search_by_description(self, query: str) -> 'GlyphExplorer':
        """
        Search glyphs by description text.
        
        Parameters:
        -----------
        query : str
            Search query
            
        Returns:
        --------
        GlyphExplorer
            Self, for method chaining
        """
        self.filtered_glyphs = [
            g for g in self.filtered_glyphs
            if g.description and query.lower() in g.description.lower()
        ]
        self._update_connections()
        return self
    
    def find_central_glyphs(self, top_n: int = 5) -> List[Glyph]:
        """
        Find central glyphs based on connection count.
        
        Parameters:
        -----------
        top_n : int
            Number of top central glyphs to return
            
        Returns:
        --------
        List[Glyph]
            Top central glyphs
        """
        # Count connections for each glyph
        glyph_ids = [g.id for g in self.filtered_glyphs]
        connection_counts = {}
        
        for glyph_id in glyph_ids:
            count = sum(
                1 for c in self.filtered_connections
                if c.source_id == glyph_id or c.target_id == glyph_id
            )
            connection_counts[glyph_id] = count
        
        # Get top N glyphs by connection count
        top_glyph_ids = sorted(
            connection_counts.keys(),
            key=lambda x: connection_counts[x],
            reverse=True
        )[:top_n]
        
        # Find corresponding glyphs
        top_glyphs = [
            g for g in self.filtered_glyphs
            if g.id in top_glyph_ids
        ]
        
        return top_glyphs
    
    def find_clusters(self, min_size: int = 3) -> Dict[str, List[Glyph]]:
        """
        Find clusters of connected glyphs.
        
        Parameters:
        -----------
        min_size : int
            Minimum cluster size
            
        Returns:
        --------
        Dict[str, List[Glyph]]
            Dictionary of clusters
        """
        # Create networkx graph
        G = nx.Graph()
        
        # Add nodes
        for glyph in self.filtered_glyphs:
            G.add_node(glyph.id)
        
        # Add edges
        for conn in self.filtered_connections:
            if conn.source_id in G and conn.target_id in G:
                G.add_edge(conn.source_id, conn.target_id, weight=conn.strength)
        
        # Find connected components (clusters)
        components = list(nx.connected_components(G))
        
        # Filter by minimum size
        clusters = {}
        for i, component in enumerate(components):
            if len(component) >= min_size:
                cluster_glyphs = [
                    g for g in self.filtered_glyphs
                    if g.id in component
                ]
                clusters[f"cluster_{i}"] = cluster_glyphs
        
        return clusters
    
    def calculate_statistics(self) -> Dict[str, Any]:
        """
        Calculate statistics for the filtered glyph map.
        
        Returns:
        --------
        Dict[str, Any]
            Dictionary of statistics
        """
        stats = {
            "num_glyphs": len(self.filtered_glyphs),
            "num_connections": len(self.filtered_connections),
            "glyph_types": {},
            "connection_types": {},
            "avg_connection_strength": 0.0,
            "glyph_size_stats": {
                "min": float('inf'),
                "max": 0.0,
                "avg": 0.0
            }
        }
        
        # Count glyph types
        for glyph in self.filtered_glyphs:
            glyph_type = glyph.type.value
            if glyph_type not in stats["glyph_types"]:
                stats["glyph_types"][glyph_type] = 0
            stats["glyph_types"][glyph_type] += 1
            
            # Update size stats
            stats["glyph_size_stats"]["min"] = min(stats["glyph_size_stats"]["min"], glyph.size)
            stats["glyph_size_stats"]["max"] = max(stats["glyph_size_stats"]["max"], glyph.size)
            stats["glyph_size_stats"]["avg"] += glyph.size
        
        if self.filtered_glyphs:
            stats["glyph_size_stats"]["avg"] /= len(self.filtered_glyphs)
        else:
            stats["glyph_size_stats"]["min"] = 0.0
        
        # Count connection types
        total_strength = 0.0
        for conn in self.filtered_connections:
            if conn.type not in stats["connection_types"]:
                stats["connection_types"][conn.type] = 0
            stats["connection_types"][conn.type] += 1
            total_strength += conn.strength
        
        if self.filtered_connections:
            stats["avg_connection_strength"] = total_strength / len(self.filtered_connections)
        
        return stats
    
    def reset_filters(self) -> 'GlyphExplorer':
        """
        Reset all filters.
        
        Returns:
        --------
        GlyphExplorer
            Self, for method chaining
        """
        self.filtered_glyphs = self.glyph_map.glyphs
        self.filtered_connections = self.glyph_map.connections
        return self
    
    def _update_connections(self):
        """Update connections based on filtered glyphs."""
        filtered_glyph_ids = [g.id for g in self.filtered_glyphs]
        self.filtered_connections = [
            c for c in self.glyph_map.connections
            if c.source_id in filtered_glyph_ids and c.target_id in filtered_glyph_ids
        ]


# Main execution for CLI usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Glyph Mapper for Attribution and Residue Visualization")
    parser.add_argument("--input", "-i", type=str, help="Input attribution or residue file")
    parser.add_argument("--output", "-o", type=str, help="Output visualization file")
    parser.add_argument("--type", "-t", type=str, default="attribution", choices=["attribution", "residue", "attention", "recursive"], help="Type of input data")
    parser.add_argument("--layout", "-l", type=str, default="force_directed", choices=["force_directed", "hierarchical", "circular", "grid", "radial"], help="Layout type")
    parser.add_argument("--width", "-w", type=int, default=1200, help="Visualization width")
    parser.add_argument("--height", "-h", type=int, default=900,
parser.add_argument("--height", "-h", type=int, default=900, help="Visualization height")
    parser.add_argument("--focus", "-f", type=str, help="Comma-separated tokens to focus on")
    parser.add_argument("--include-tokens", action="store_true", help="Include token sentinels in visualization")
    parser.add_argument("--cluster", action="store_true", help="Apply clustering to similar patterns")
    parser.add_argument("--save-map", "-s", type=str, help="Save glyph map to file")
    parser.add_argument("--interactive", "-i", action="store_true", help="Generate interactive visualization")
    
    args = parser.parse_args()
    
    # Initialize mapper
    mapper = GlyphMapper()
    
    if args.input:
        # Load input data
        with open(args.input, "r") as f:
            data = json.load(f)
        
        # Process based on type
        if args.type == "attribution":
            # Convert to AttributionMap
            attribution_map = AttributionMap(
                prompt_tokens=data.get("prompt_tokens", []),
                output_tokens=data.get("output_tokens", []),
                links=[
                    AttributionLink(
                        source_idx=link.get("source_idx", 0),
                        target_idx=link.get("target_idx", 0),
                        attribution_type=AttributionType(link.get("attribution_type", "direct")),
                        strength=link.get("strength", 0.5),
                        attention_heads=link.get("attention_heads", []),
                        layers=link.get("layers", []),
                        intermediate_tokens=link.get("intermediate_tokens", []),
                        residue=link.get("residue")
                    )
                    for link in data.get("links", [])
                ],
                token_salience=data.get("token_salience", {}),
                attribution_gaps=data.get("attribution_gaps", []),
                collapsed_regions=data.get("collapsed_regions", []),
                uncertainty=data.get("uncertainty", {}),
                metadata=data.get("metadata", {})
            )
            
            # Parse focus tokens if provided
            focus_on = args.focus.split(",") if args.focus else None
            
            # Create glyph map
            glyph_map = mapper.map_attribution(
                attribution_map=attribution_map,
                layout_type=args.layout,
                dimensions=(args.width, args.height),
                include_tokens=args.include_tokens,
                focus_on=focus_on
            )
        
        elif args.type == "residue":
            # Convert to ResiduePattern list
            residue_patterns = [
                ResiduePattern(
                    type=pattern.get("type", "unknown"),
                    pattern=pattern.get("pattern", ""),
                    context=pattern.get("context", {}),
                    signature=pattern.get("signature", ""),
                    confidence=pattern.get("confidence", 0.5)
                )
                for pattern in data
            ]
            
            # Create glyph map
            glyph_map = mapper.map_residue_patterns(
                residue_patterns=residue_patterns,
                layout_type=args.layout,
                dimensions=(args.width, args.height),
                cluster_patterns=args.cluster
            )
        
        elif args.type == "attention":
            # Create glyph map
            glyph_map = mapper.map_attention_heads(
                attention_data=data,
                layout_type=args.layout,
                dimensions=(args.width, args.height),
                include_tokens=args.include_tokens
            )
        
        elif args.type == "recursive":
            # Create glyph map
            glyph_map = mapper.map_recursive_trace(
                trace_data=data,
                layout_type=args.layout,
                dimensions=(args.width, args.height)
            )
        
        else:
            print(f"Unknown data type: {args.type}")
            exit(1)
        
        # Save glyph map if requested
        if args.save_map:
            mapper.save_glyph_map(glyph_map, args.save_map)
        
        # Generate visualization
        if args.output:
            mapper.visualize(
                glyph_map=glyph_map,
                output_path=args.output,
                interactive=args.interactive
            )
            print(f"Visualization saved to {args.output}")
        else:
            # Display basic statistics
            explorer = GlyphExplorer(glyph_map)
            stats = explorer.calculate_statistics()
            
            print(f"Glyph Map Statistics:")
            print(f"  Number of glyphs: {stats['num_glyphs']}")
            print(f"  Number of connections: {stats['num_connections']}")
            print(f"  Glyph types: {stats['glyph_types']}")
            print(f"  Connection types: {stats['connection_types']}")
            print(f"  Average connection strength: {stats['avg_connection_strength']:.2f}")
            
            # Show central glyphs
            central_glyphs = explorer.find_central_glyphs(top_n=3)
            print(f"\nCentral Glyphs:")
            for glyph in central_glyphs:
                print(f"  {glyph.symbol} - {glyph.description}")
    else:
        print("No input file specified. Use --input to provide input data.")
        exit(1)
