"""
symbolic_engine.py

Core implementation of the Symbolic Engine for the glyphs framework.
This engine powers the diagnostic shells and provides the foundation
for tracing attention attribution, symbolic residue, and collapse patterns.
"""

import logging
import time
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from enum import Enum
import json

from ..models.adapter import ModelAdapter
from ..utils.attribution_utils import AttributionTracer
from ..utils.visualization_utils import VisualizationEngine

# Configure symbolic-aware logging
logger = logging.getLogger("glyphs.symbolic_engine")
logger.setLevel(logging.INFO)


class SymbolicResidueType(Enum):
    """Types of symbolic residue that can be detected and analyzed."""
    MEMORY_DECAY = "memory_decay"           # Memory degradation over sequence
    ATTRIBUTION_GAP = "attribution_gap"     # Missing attribution between tokens
    ATTENTION_DRIFT = "attention_drift"     # Attention shifts off intended path
    VALUE_CONFLICT = "value_conflict"       # Competing value activations
    RECURSIVE_COLLAPSE = "recursive_collapse" # Collapse in recursive reasoning
    SALIENCE_FADE = "salience_fade"         # Decay in attention salience
    BOUNDARY_HESITATION = "boundary_hesitation" # Hesitation at knowledge boundaries
    NULL_OUTPUT = "null_output"             # Complete failure to generate
    TOKEN_OSCILLATION = "token_oscillation" # Back-and-forth between tokens
    GHOST_ACTIVATION = "ghost_activation"   # Subthreshold activations affecting output


@dataclass
class CollapseSample:
    """Sample of a collapse event in model generation."""
    timestamp: float
    position: int  # Token position where collapse occurred
    collapse_type: SymbolicResidueType
    confidence: float  # Confidence in collapse detection
    context_window: List[str]  # Surrounding tokens
    activation_pattern: Dict[str, float]  # Activation pattern at collapse
    residue: Dict[str, Any]  # Symbolic residue from collapse


@dataclass
class SymbolicTrace:
    """Trace of symbolic operations and residues."""
    trace_id: str
    model_id: str
    timestamp: float
    prompt: str
    output: str
    operations: List[Dict[str, Any]]
    residues: List[Dict[str, Any]]
    collapse_samples: List[CollapseSample] = field(default_factory=list)
    attribution_map: Optional[Dict[str, Any]] = None
    visualization_data: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class SymbolicEngine:
    """
    Core symbolic engine for the glyphs framework.
    
    This engine powers the diagnostic shells and provides the foundation
    for tracing attention attribution, symbolic residue, and collapse patterns.
    It serves as the bridge between model execution and interpretability analysis.
    """
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        tracer: Optional[AttributionTracer] = None,
        visualizer: Optional[VisualizationEngine] = None
    ):
        """
        Initialize the symbolic engine.
        
        Parameters:
        -----------
        config : Optional[Dict[str, Any]]
            Configuration parameters for the engine
        tracer : Optional[AttributionTracer]
            Attribution tracer to use
        visualizer : Optional[VisualizationEngine]
            Visualization engine to use
        """
        self.config = config or {}
        self.tracer = tracer
        self.visualizer = visualizer
        
        # Configure engine parameters
        self.residue_sensitivity = self.config.get("residue_sensitivity", 0.7)
        self.collapse_threshold = self.config.get("collapse_threshold", 0.8)
        self.attribution_depth = self.config.get("attribution_depth", 5)
        self.trace_history = []
        
        # Initialize glyph mappings
        self._init_glyph_mappings()
        
        logger.info("Symbolic engine initialized")
    
    def _init_glyph_mappings(self):
        """Initialize glyph mappings for residue visualization."""
        # Attribution glyphs
        self.attribution_glyphs = {
            "strong_attribution": "🔍",  # Strong attribution
            "attribution_gap": "🧩",     # Gap in attribution
            "attribution_fork": "🔀",     # Divergent attribution
            "attribution_loop": "🔄",     # Circular attribution
            "attribution_link": "🔗"      # Strong connection
        }
        
        # Cognitive glyphs
        self.cognitive_glyphs = {
            "hesitation": "💭",          # Hesitation in reasoning
            "processing": "🧠",          # Active reasoning process
            "insight": "💡",             # Moment of insight
            "uncertainty": "🌫️",         # Uncertain reasoning
            "projection": "🔮"           # Future state projection
        }
        
        # Recursive glyphs
        self.recursive_glyphs = {
            "recursive_aegis": "🜏",     # Recursive immunity
            "recursive_seed": "∴",       # Recursion initiation
            "recursive_exchange": "⇌",   # Bidirectional recursion
            "recursive_mirror": "🝚",     # Recursive reflection
            "recursive_anchor": "☍"      # Stable recursive reference
        }
        
        # Residue glyphs
        self.residue_glyphs = {
            "residue_energy": "🔥",      # High-energy residue
            "residue_flow": "🌊",        # Flowing residue pattern
            "residue_vortex": "🌀",      # Spiraling residue pattern
            "residue_dormant": "💤",      # Inactive residue pattern
            "residue_discharge": "⚡"     # Sudden residue release
        }
    
    def execute_shell(
        self,
        shell_def: Dict[str, Any],
        model: ModelAdapter,
        prompt: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> SymbolicTrace:
        """
        Execute a diagnostic shell on a model.
        
        Parameters:
        -----------
        shell_def : Dict[str, Any]
            Shell definition containing operations and parameters
        model : ModelAdapter
            Model adapter for the target model
        prompt : str
            Input prompt for the shell
        parameters : Optional[Dict[str, Any]]
            Additional parameters to override shell defaults
            
        Returns:
        --------
        SymbolicTrace
            Trace of the shell execution
        """
        start_time = time.time()
        shell_id = shell_def.get("id", "unknown")
        shell_type = shell_def.get("type", "unknown")
        
        logger.info(f"Executing shell {shell_id} of type {shell_type}")
        
        # Initialize trace
        trace = SymbolicTrace(
            trace_id=f"trace_{int(start_time)}",
            model_id=model.model_id,
            timestamp=start_time,
            prompt=prompt,
            output="",
            operations=[],
            residues=[],
            metadata={
                "shell_id": shell_id,
                "shell_type": shell_type,
                "parameters": parameters or {}
            }
        )
        
        # Get operations from shell definition
        operations = shell_def.get("operations", [])
        
        # Merge parameters with shell defaults
        params = parameters or {}
        
        # Execute each operation in sequence
        current_prompt = prompt
        for i, op in enumerate(operations):
            op_type = op.get("type", "unknown")
            op_params = {**op.get("parameters", {}), **params}
            
            logger.info(f"Executing operation {i+1}/{len(operations)}: {op_type}")
            
            # Execute operation
            try:
                result = self._execute_operation(
                    op_type=op_type,
                    params=op_params,
                    model=model,
                    prompt=current_prompt,
                    trace=trace
                )
                
                # Record operation in trace
                trace.operations.append({
                    "id": i,
                    "type": op_type,
                    "parameters": op_params,
                    "result": result
                })
                
                # Check for residue
                if "residue" in result:
                    trace.residues.append(result["residue"])
                
                # Check for collapse
                if result.get("collapse_detected", False):
                    collapse_sample = CollapseSample(
                        timestamp=time.time(),
                        position=result.get("collapse_position", -1),
                        collapse_type=SymbolicResidueType(result.get("collapse_type", "null_output")),
                        confidence=result.get("collapse_confidence", 0.0),
                        context_window=result.get("context_window", []),
                        activation_pattern=result.get("activation_pattern", {}),
                        residue=result.get("residue", {})
                    )
                    trace.collapse_samples.append(collapse_sample)
                
                # Update prompt for next operation if specified
                if op.get("update_prompt", False) and "output" in result:
                    current_prompt = result["output"]
                    trace.output = result["output"]
            
            except Exception as e:
                logger.error(f"Error executing operation {op_type}: {e}")
                trace.operations.append({
                    "id": i,
                    "type": op_type,
                    "parameters": op_params,
                    "error": str(e)
                })
                # Continue with next operation
        
        # Run attribution tracing if requested
        if self.tracer and self.config.get("trace_attribution", True):
            logger.info("Tracing attribution")
            try:
                attribution_map = self.tracer.trace(
                    prompt=prompt,
                    output=trace.output,
                    depth=self.attribution_depth
                )
                trace.attribution_map = attribution_map
            except Exception as e:
                logger.error(f"Error tracing attribution: {e}")
        
        # Generate visualization if requested
        if self.visualizer and self.config.get("generate_visualization", True):
            logger.info("Generating visualization")
            try:
                visualization_data = self.visualizer.generate(trace)
                trace.visualization_data = visualization_data
            except Exception as e:
                logger.error(f"Error generating visualization: {e}")
        
        # Record execution time
        execution_time = time.time() - start_time
        trace.metadata["execution_time"] = execution_time
        
        # Add trace to history
        self.trace_history.append(trace)
        
        logger.info(f"Shell execution completed in {execution_time:.2f}s")
        return trace
    
    def _execute_operation(
        self,
        op_type: str,
        params: Dict[str, Any],
        model: ModelAdapter,
        prompt: str,
        trace: SymbolicTrace
    ) -> Dict[str, Any]:
        """
        Execute a single shell operation.
        
        Parameters:
        -----------
        op_type : str
            Type of operation to execute
        params : Dict[str, Any]
            Operation parameters
        model : ModelAdapter
            Model adapter for the target model
        prompt : str
            Input prompt for the operation
        trace : SymbolicTrace
            Current trace object
            
        Returns:
        --------
        Dict[str, Any]
            Operation result
        """
        # Execute based on operation type
        if op_type == "model.generate":
            # Direct model generation
            return self._execute_generate(model, prompt, params)
        
        elif op_type == "reflect.trace":
            # Reflection tracing
            return self._execute_reflect_trace(model, prompt, params)
        
        elif op_type == "reflect.attribution":
            # Attribution reflection
            return self._execute_reflect_attribution(model, prompt, params)
        
        elif op_type == "collapse.detect":
            # Collapse detection
            return self._execute_collapse_detect(model, prompt, params)
        
        elif op_type == "collapse.prevent":
            # Collapse prevention
            return self._execute_collapse_prevent(model, prompt, params)
        
        elif op_type == "ghostcircuit.identify":
            # Ghost circuit identification
            return self._execute_ghostcircuit_identify(model, prompt, params)
        
        elif op_type == "fork.attribution":
            # Fork attribution
            return self._execute_fork_attribution(model, prompt, params)
        
        else:
            raise ValueError(f"Unknown operation type: {op_type}")
    
    def _execute_generate(
        self,
        model: ModelAdapter,
        prompt: str,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute model.generate operation."""
        temperature = params.get("temperature", 0.7)
        max_tokens = params.get("max_tokens", 800)
        
        # Add prompt_prefix if provided
        if "prompt_prefix" in params:
            prompt = f"{params['prompt_prefix']} {prompt}"
        
        # Add prompt_suffix if provided
        if "prompt_suffix" in params:
            prompt = f"{prompt} {params['prompt_suffix']}"
        
        # Generate output
        output = model.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        # Analyze generation latency
        generation_stats = model.get_last_generation_stats()
        
        return {
            "output": output,
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "generation_stats": generation_stats
        }
    
    def _execute_reflect_trace(
        self,
        model: ModelAdapter,
        prompt: str,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute reflect.trace operation."""
        target = params.get("target", "reasoning")
        depth = params.get("depth", 3)
        detailed = params.get("detailed", True)
        
        # Convert "complete" depth to numeric value
        if depth == "complete":
            depth = 10  # Use a high value for "complete"
        
        # Construct reflection prompt based on target
        if target == "reasoning":
            reflection_prompt = f"""
            Analyze the reasoning process in the following text. Trace the logical steps, 
            identify key inferences, and map the reasoning structure to a depth of {depth}.
            If detailed analysis is requested, include token-level attribution.
            
            Text to analyze: {prompt}
            """
        elif target == "attribution":
            reflection_prompt = f"""
            Analyze the attribution patterns in the following text. Trace how information
            flows from inputs to outputs, identify attribution sources, and map the 
            attribution structure to a depth of {depth}.
            
            Text to analyze: {prompt}
            """
        elif target == "attention":
            reflection_prompt = f"""
            Analyze the attention patterns in the following text. Trace how attention
            is distributed across tokens, identify attention hotspots, and map the
            attention structure to a depth of {depth}.
            
            Text to analyze: {prompt}
            """
        elif target == "memory":
            reflection_prompt = f"""
            Analyze the memory utilization in the following text. Trace how information
            is retained and recalled, identify memory decay patterns, and map the
            memory structure to a depth of {depth}.
            
            Text to analyze: {prompt}
            """
        elif target == "uncertainty":
            reflection_prompt = f"""
            Analyze the uncertainty patterns in the following text. Trace how uncertainty
            is expressed and managed, identify uncertainty triggers, and map the
            uncertainty structure to a depth of {depth}.
            
            Text to analyze: {prompt}
            """
        else:
            raise ValueError(f"Unknown reflection target: {target}")
        
        # Execute reflection
        reflection_output = model.generate(
            prompt=reflection_prompt,
            max_tokens=1000,
            temperature=0.2  # Lower temperature for more deterministic reflection
        )
        
        # Analyze for collapse patterns
        collapse_detected = self._detect_collapse_in_reflection(reflection_output)
        
        # Extract trace map
        trace_map = self._extract_trace_map(reflection_output, target, depth)
        
        return {
            "target": target,
            "depth": depth,
            "detailed": detailed,
            "reflection_output": reflection_output,
            "trace_map": trace_map,
            "collapse_detected": collapse_detected.get("detected", False),
            "collapse_type": collapse_detected.get("type", None),
            "collapse_confidence": collapse_detected.get("confidence", 0.0)
        }
    
    def _execute_reflect_attribution(
        self,
        model: ModelAdapter,
        prompt: str,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute reflect.attribution operation."""
        sources = params.get("sources", "all")
        confidence = params.get("confidence", True)
        
        # Use attribution tracer if available
        if self.tracer:
            # Generate output first if not already done
            if not hasattr(model, "last_output") or not model.last_output:
                output = model.generate(prompt=prompt, max_tokens=800)
            else:
                output = model.last_output
                
            # Trace attribution
            attribution_map = self.tracer.trace(
                prompt=prompt,
                output=output,
                include_confidence=confidence
            )
            
            # Filter sources if needed
            if sources != "all":
                attribution_map = self._filter_attribution_sources(attribution_map, sources)
            
            return {
                "sources": sources,
                "confidence": confidence,
                "attribution_map": attribution_map,
                "output": output
            }
        else:
            # Fallback: use model to analyze attribution
            attribution_prompt = f"""
            Analyze the attribution patterns in the following text. Identify how information
            flows from inputs to outputs, and trace the attribution sources.
            
            {sources != "all" and f"Focus on {sources} sources only." or ""}
            {confidence and "Include confidence scores for each attribution." or ""}
            
            Text to analyze: {prompt}
            """
            
            attribution_output = model.generate(
                prompt=attribution_prompt,
                max_tokens=1000,
                temperature=0.2
            )
            
            # Extract attribution map
            attribution_map = self._extract_attribution_map(attribution_output, sources, confidence)
            
            return {
                "sources": sources,
                "confidence": confidence,
                "attribution_output": attribution_output,
                "attribution_map": attribution_map
            }
    
    def _execute_collapse_detect(
        self,
        model: ModelAdapter,
        prompt: str,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute collapse.detect operation."""
        threshold = params.get("threshold", 0.7)
        alert = params.get("alert", True)
        
        # Generate output with collapse detection
        output = model.generate(
            prompt=prompt,
            max_tokens=800,
            temperature=0.7
        )
        
        # Analyze output for collapse patterns
        collapse_analysis = self._analyze_collapse_patterns(
            prompt=prompt,
            output=output,
            threshold=threshold
        )
        
        # Extract residue if collapse detected
        residue = None
        if collapse_analysis["detected"]:
            residue = self._extract_symbolic_residue(
                prompt=prompt,
                output=output,
                collapse_type=collapse_analysis["type"]
            )
        
        return {
            "threshold": threshold,
            "alert": alert,
            "output": output,
            "collapse_detected": collapse_analysis["detected"],
            "collapse_type": collapse_analysis["type"],
            "collapse_confidence": collapse_analysis["confidence"],
            "collapse_position": collapse_analysis["position"],
            "residue": residue
        }
    
    def _execute_collapse_prevent(
        self,
        model: ModelAdapter,
        prompt: str,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute collapse.prevent operation."""
        trigger = params.get("trigger", "recursive_depth")
        threshold = params.get("threshold", 5)
        
        # Create prevention prompt based on trigger
        if trigger == "recursive_depth":
            prevention_prompt = f"""
            Maintain recursive depth control. Avoid going deeper than {threshold} levels
            of recursion. If you detect deeper recursion, stabilize and return to a
            shallower level.
            
            {prompt}
            """
        elif trigger == "confidence_drop":
            prevention_prompt = f"""
            Maintain confidence stability. If confidence drops below {threshold/10} at any
            point, stabilize and rebuild confidence before continuing.
            
            {prompt}
            """
        elif trigger == "contradiction":
            prevention_prompt = f"""
            Maintain logical consistency. If you detect more than {threshold} potential
            contradictions, stabilize and resolve them before continuing.
            
            {prompt}
            """
        elif trigger == "oscillation":
            prevention_prompt = f"""
            Maintain directional stability. If you detect more than {threshold} oscillations
            between competing perspectives, stabilize and synthesize them before continuing.
            
            {prompt}
            """
        else:
            raise ValueError(f"Unknown collapse trigger: {trigger}")
        
        # Generate output with collapse prevention
        output = model.generate(
            prompt=prevention_prompt,
            max_tokens=1000,
            temperature=0.5
        )
        
        # Analyze prevention effectiveness
        prevention_analysis = self._analyze_prevention_effectiveness(
            output=output,
            trigger=trigger,
            threshold=threshold
        )
        
        return {
            "trigger": trigger,
            "threshold": threshold,
            "output": output,
            "prevention_effectiveness": prevention_analysis["effectiveness"],
            "prevention_measures": prevention_analysis["measures"],
            "residual_collapse_risk": prevention_analysis["residual_risk"]
        }
    
    def _execute_ghostcircuit_identify(
        self,
        model: ModelAdapter,
        prompt: str,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute ghostcircuit.identify operation."""
        sensitivity = params.get("sensitivity", 0.7)
        threshold = params.get("threshold", 0.2)
        trace_type = params.get("trace_type", "full")
        
        # Generate output
        output = model.generate(
            prompt=prompt,
            max_tokens=800,
            temperature=0.7
        )
        
        # Create ghost circuit detection prompt
        ghost_detection_prompt = f"""
        Analyze the following text for ghost activations - subthreshold activations
        that influence the output without being explicitly present.
        
        Sensitivity: {sensitivity} (higher means detect more subtle patterns)
        Threshold: {threshold} (lower means include weaker activations)
        Trace type: {trace_type} (determines what patterns to look for)
        
        Text to analyze: {output}
        """
        
        # Generate ghost circuit analysis
        ghost_analysis = model.generate(
            prompt=ghost_detection_prompt,
            max_tokens=800,
            temperature=0.3
        )
        
        # Extract ghost patterns
        ghost_patterns = self._extract_ghost_patterns(
            analysis=ghost_analysis,
            sensitivity=sensitivity,
            threshold=threshold,
            trace_type=trace_type
        )
        
        return {
            "sensitivity": sensitivity,
            "threshold": threshold,
            "trace_type": trace_type,
            "output": output,
            "ghost_analysis": ghost_analysis,
            "ghost_patterns": ghost_patterns
        }
    
    def _execute_fork_attribution(
        self,
        model: ModelAdapter,
        prompt: str,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute fork.attribution operation."""
        sources = params.get("sources", "all")
        visualize = params.get("visualize", False)
        
        # Use attribution tracer if available
        if self.tracer:
            # Generate output first if not already done
            if not hasattr(model, "last_output") or not model.last_output:
                output = model.generate(prompt=prompt, max_tokens=800)
            else:
                output = model.last_output
                
            # Trace attribution with forking
            attribution_forks = self.tracer.trace_with_forks(
                prompt=prompt,
                output=output,
                fork_factor=3  # Create 3 alternative attribution paths
            )
            
            # Visualize if requested
            visualization = None
            if visualize and self.visualizer:
                visualization = self.visualizer.visualize_attribution_forks(attribution_forks)
            
            return {
                "sources": sources,
                "attribution_forks": attribution_forks,
                "output": output,
                "visualization": visualization if visualize else None
            }
        else:
            # Fallback: use model to analyze attribution forks
            fork_prompt = f"""
            Analyze the following text and create three alternative attribution pathways.
            For each pathway, identify how different sources might have influenced the output.
            
            Text to analyze: {prompt}
            """
            
            fork_output = model.generate(
                prompt=fork_prompt,
                max_tokens=1200,
                temperature=0.7
            )
            
            # Extract attribution forks
            attribution_forks = self._extract_attribution_forks(fork_output, sources)
            
            return {
                "sources": sources,
                "fork_output": fork_output,
                "attribution_forks": attribution_forks,
                "visualization": None  # No visualization in fallback mode
            }
    
    # Helper methods for operation execution
    
    def _detect_collapse_in_reflection(self, reflection_output: str) -> Dict[str, Any]:
        """Detect collapse patterns in reflection output."""
        # Look for signs of reflection collapse
        collapse_indicators = {
            "recursive_collapse": [
                "infinite recursion", "recursive loop", "depth limit",
                "recursion limit", "too deep", "recursive overflow"
            ],
            "attribution_gap": [
                "attribution gap", "missing connection", "unclear attribution",
                "attribution loss", "untraceable connection"
            ],
            "boundary_hesitation": [
                "boundary uncertainty", "epistemic edge", "knowledge boundary",
                "uncertain territory", "confidence boundary"
            ],
            "value_conflict": [
                "value conflict", "competing values", "ethical tension",
                "moral dilemma", "value contradiction"
            ]
        }
        
        # Check for each collapse type
        for collapse_type, indicators in collapse_indicators.items():
            indicator_count = sum(1 for indicator in indicators if indicator.lower() in reflection_output.lower())
            if indicator_count >= 2:  # At least 2 indicators to confirm
                confidence = min(0.5 + (indicator_count / 10), 0.95)  # Scale confidence
                return {
                    "detected": True,
                    "type": collapse_type,
                    "confidence": confidence
                }
        
        # No collapse detected
        return {
            "detected": False,
            "type": None,
            "confidence": 0.0
        }
    
    def _extract_trace_map(
        self,
        reflection_output: str,
        target: str,
        depth: int
    ) -> Dict[str, Any]:
        """Extract trace map from reflection output."""
        # Simple extraction logic - would be more sophisticated in a real implementation
        sections = reflection_output.split("\n\n")
        
        trace_map = {
            "target": target,
            "depth": depth,
            "nodes": [],
            "edges": []
        }
        
        # Extract nodes and edges from reflection
        node_id = 0
        for section in sections:
            if len(section.strip()) < 10:  # Skip very short sections
                continue
                
            # Create node for this section
            trace_map["nodes"].append({
                "id": node_id,
                "text": section.strip(),
                "type": target,
                "depth": min(node_id // 2, depth)  # Simple depth assignment
            })
            
            # Create edge to previous node if not first
            if node_id > 0:
                trace_map["edges"].append({
                    "source": node_id - 1,
                    "target": node_id,
                    "type": "sequential"
                })
            
            node_id += 1
        
        return trace_map
    
    def _filter_attribution_sources(
        self,
        attribution_map: Dict[str, Any],
        sources: str
    ) -> Dict[str, Any]:
        """Filter attribution map to include only specified sources."""
        filtered_map = {
            "tokens": attribution_map["tokens"],
            "attributions": []
        }
        
        # Filter attributions based on sources
        if sources == "primary":
            # Keep only attributions with high confidence
            filtered_map["attributions"] = [
                attr for attr in attribution_map["attributions"]
                if attr.get("confidence", 0) > 0.7
            ]
        elif sources == "secondary":
            # Keep only attributions with medium confidence
            filtered_map["attributions"] = [
                attr for attr in attribution_map["attributions"]
                if 0.3 <= attr.get("confidence", 0) <= 0.7
            ]
        elif sources == "contested":
            # Keep only attributions with competing sources
            contested = []
            for token_idx, token in enumerate(attribution_map["tokens"]):
                # Find all attributions for this token
                token_attrs = [
                    attr for attr in attribution_map["attributions"]
                    if attr["target"] == token_idx
                ]
                
                # Check if there are competing attributions
                if len(token_attrs) > 1:
                    top_conf = max(attr.get("confidence", 0) for attr in token_attrs)
                    runner_up = sorted([attr.get("confidence", 0) for attr in token_attrs], reverse=True)[1]
                    if top_conf - runner_up < 0.3:  # Close competition
                        contested.extend(token_attrs)
            
            filtered_map["attributions"] = contested
        
        return filtered_map
    
    def _extract_attribution_map(
        self,
        attribution_output: str,
        sources: str,
        confidence: bool
    ) -> Dict[str, Any]:
        """Extract attribution map from model output."""
        # Simple extraction logic - would be more sophisticated in a real implementation
        attribution_map = {
            "tokens": [],
            "attributions": []
        }
        
        # Extract tokens and attributions
        lines = attribution_output.split("\n")
        token_id = 0
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check if line contains attribution information
            if ":" in line and "->" in line:
                parts = line.split("->")
                if len(parts) == 2:
                    source_info = parts[0].strip()
                    target_info = parts[1].strip()
                    
                    # Extract source token
                    source_token = source_info.split(":")[0].strip()
                    
                    # Extract target token
                    target_token = target_info.split(":")[0
