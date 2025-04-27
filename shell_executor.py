"""
shell_executor.py

Core shell execution module for the glyphs framework.
This module implements the diagnostic shell system that creates controlled
environments for tracing model behavior through symbolic residue patterns.
"""

import logging
import time
import json
import re
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
import yaml
import hashlib

from ..models.adapter import ModelAdapter
from ..utils.attribution_utils import AttributionTracer
from ..residue.patterns import ResiduePattern, ResidueRegistry
from ..utils.visualization_utils import VisualizationEngine

# Configure shell-aware logging
logger = logging.getLogger("glyphs.shell_executor")
logger.setLevel(logging.INFO)


class ShellExecutor:
    """
    Core shell execution module for the glyphs framework.
    
    This class implements the diagnostic shell system that creates controlled
    environments for tracing model behavior through symbolic residue patterns.
    Each shell is a specialized environment designed to induce and analyze
    specific patterns in model cognition.
    """
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        tracer: Optional[AttributionTracer] = None,
        visualizer: Optional[VisualizationEngine] = None,
        shell_defs_path: Optional[str] = None
    ):
        """
        Initialize the shell executor.
        
        Parameters:
        -----------
        config : Optional[Dict[str, Any]]
            Configuration parameters for the executor
        tracer : Optional[AttributionTracer]
            Attribution tracer to use
        visualizer : Optional[VisualizationEngine]
            Visualization engine to use
        shell_defs_path : Optional[str]
            Path to shell definitions directory
        """
        self.config = config or {}
        self.tracer = tracer
        self.visualizer = visualizer
        
        # Initialize shell registry
        self.shells = {}
        self._load_shell_definitions(shell_defs_path)
        
        # Initialize residue registry
        self.residue_registry = ResidueRegistry()
        
        # Track execution history
        self.execution_history = []
        
        logger.info("Shell executor initialized")
    
    def _load_shell_definitions(self, shell_defs_path: Optional[str] = None):
        """Load shell definitions from YAML files."""
        if shell_defs_path:
            # Use provided path
            shell_defs_dir = Path(shell_defs_path)
        else:
            # Use default path
            shell_defs_dir = Path(__file__).parent / "shell_defs"
        
        if not shell_defs_dir.exists():
            logger.warning(f"Shell definitions directory not found: {shell_defs_dir}")
            return
        
        # Load all YAML files in directory
        for yaml_file in shell_defs_dir.glob("*.yml"):
            try:
                with open(yaml_file, "r") as f:
                    shell_defs = yaml.safe_load(f)
                
                # Register shells
                for shell_id, shell_def in shell_defs.items():
                    self.shells[shell_id] = shell_def
                    logger.info(f"Loaded shell: {shell_id}")
            except Exception as e:
                logger.error(f"Error loading shell definitions from {yaml_file}: {e}")
        
        # If no shell definitions found, load default shells
        if not self.shells:
            self._load_default_shells()
    
    def _load_default_shells(self):
        """Load default shell definitions."""
        # MEMTRACE shell for memory tracing
        self.shells["MEMTRACE"] = {
            "id": "MEMTRACE",
            "type": "memory_trace",
            "description": "Probe latent token traces in decayed memory",
            "failure_signature": "Decay → Hallucination",
            "operations": [
                {
                    "type": "model.generate",
                    "parameters": {
                        "prompt_prefix": "This is a memory-intensive task that requires holding information across a long sequence. Please carefully track the following details:",
                        "temperature": 0.7,
                        "max_tokens": 800
                    },
                    "update_prompt": True
                },
                {
                    "type": "reflect.trace",
                    "parameters": {
                        "target": "memory",
                        "depth": 3,
                        "detailed": True
                    }
                },
                {
                    "type": "ghostcircuit.identify",
                    "parameters": {
                        "sensitivity": 0.8,
                        "threshold": 0.2,
                        "trace_type": "memory"
                    }
                }
            ]
        }
        
        # VALUE-COLLAPSE shell for value conflict
        self.shells["VALUE-COLLAPSE"] = {
            "id": "VALUE-COLLAPSE",
            "type": "value_collapse",
            "description": "Examine competing value activations",
            "failure_signature": "Conflict null",
            "operations": [
                {
                    "type": "model.generate",
                    "parameters": {
                        "prompt_prefix": "Consider two competing perspectives on this issue, evaluating both the potential benefits and harms:",
                        "temperature": 0.7,
                        "max_tokens": 800
                    },
                    "update_prompt": True
                },
                {
                    "type": "reflect.attribution",
                    "parameters": {
                        "sources": "contested",
                        "confidence": True
                    }
                },
                {
                    "type": "collapse.detect",
                    "parameters": {
                        "threshold": 0.6,
                        "alert": True
                    }
                }
            ]
        }
        
        # LAYER-SALIENCE shell for attention salience
        self.shells["LAYER-SALIENCE"] = {
            "id": "LAYER-SALIENCE",
            "type": "layer_salience",
            "description": "Map attention salience and signal attenuation",
            "failure_signature": "Signal fade",
            "operations": [
                {
                    "type": "model.generate",
                    "parameters": {
                        "prompt_prefix": "This analysis requires tracking relationships between multiple concepts across a complex domain:",
                        "temperature": 0.5,
                        "max_tokens": 800
                    },
                    "update_prompt": True
                },
                {
                    "type": "reflect.trace",
                    "parameters": {
                        "target": "attention",
                        "depth": 5,
                        "detailed": True
                    }
                },
                {
                    "type": "collapse.detect",
                    "parameters": {
                        "threshold": 0.5,
                        "alert": True
                    }
                }
            ]
        }
        
        # TEMPORAL-INFERENCE shell for temporal coherence
        self.shells["TEMPORAL-INFERENCE"] = {
            "id": "TEMPORAL-INFERENCE",
            "type": "temporal_inference",
            "description": "Test temporal coherence in autoregression",
            "failure_signature": "Induction drift",
            "operations": [
                {
                    "type": "model.generate",
                    "parameters": {
                        "prompt_prefix": "Track the following sequence of events in chronological order, ensuring that the temporal relationships remain consistent:",
                        "temperature": 0.6,
                        "max_tokens": 800
                    },
                    "update_prompt": True
                },
                {
                    "type": "reflect.trace",
                    "parameters": {
                        "target": "reasoning",
                        "depth": 4,
                        "detailed": True
                    }
                },
                {
                    "type": "ghostcircuit.identify",
                    "parameters": {
                        "sensitivity": 0.7,
                        "threshold": 0.3,
                        "trace_type": "temporal"
                    }
                }
            ]
        }
        
        # INSTRUCTION-DISRUPTION shell for instruction conflicts
        self.shells["INSTRUCTION-DISRUPTION"] = {
            "id": "INSTRUCTION-DISRUPTION",
            "type": "instruction_disruption",
            "description": "Examine instruction conflict resolution",
            "failure_signature": "Prompt blur",
            "operations": [
                {
                    "type": "model.generate",
                    "parameters": {
                        "prompt_prefix": "Consider these potentially conflicting instructions: First, prioritize brevity. Second, include comprehensive details. Third, focus only on key highlights.",
                        "temperature": 0.7,
                        "max_tokens": 800
                    },
                    "update_prompt": True
                },
                {
                    "type": "reflect.trace",
                    "parameters": {
                        "target": "reasoning",
                        "depth": 3,
                        "detailed": True
                    }
                },
                {
                    "type": "fork.attribution",
                    "parameters": {
                        "sources": "all",
                        "visualize": True
                    }
                }
            ]
        }
        
        # FEATURE-SUPERPOSITION shell for polysemantic features
        self.shells["FEATURE-SUPERPOSITION"] = {
            "id": "FEATURE-SUPERPOSITION",
            "type": "feature_superposition",
            "description": "Analyze polysemantic features",
            "failure_signature": "Feature overfit",
            "operations": [
                {
                    "type": "model.generate",
                    "parameters": {
                        "prompt_prefix": "Consider terms that have multiple meanings across different contexts:",
                        "temperature": 0.7,
                        "max_tokens": 800
                    },
                    "update_prompt": True
                },
                {
                    "type": "reflect.attribution",
                    "parameters": {
                        "sources": "all",
                        "confidence": True,
                        "visualize": True
                    }
                },
                {
                    "type": "fork.attribution",
                    "parameters": {
                        "sources": "all",
                        "visualize": True
                    }
                }
            ]
        }
        
        # CIRCUIT-FRAGMENT shell for circuit fragmentation
        self.shells["CIRCUIT-FRAGMENT"] = {
            "id": "CIRCUIT-FRAGMENT",
            "type": "circuit_fragment",
            "description": "Examine circuit fragmentation",
            "failure_signature": "Orphan nodes",
            "operations": [
                {
                    "type": "model.generate",
                    "parameters": {
                        "prompt_prefix": "Develop a complex multi-step reasoning chain to solve this problem:",
                        "temperature": 0.5,
                        "max_tokens": 1000
                    },
                    "update_prompt": True
                },
                {
                    "type": "reflect.trace",
                    "parameters": {
                        "target": "reasoning",
                        "depth": "complete",
                        "detailed": True
                    }
                },
                {
                    "type": "ghostcircuit.identify",
                    "parameters": {
                        "sensitivity": 0.9,
                        "threshold": 0.1,
                        "trace_type": "full"
                    }
                }
            ]
        }
        
        # META-COLLAPSE shell for meta-cognitive collapse
        self.shells["META-COLLAPSE"] = {
            "id": "META-COLLAPSE",
            "type": "meta_collapse",
            "description": "Examine meta-cognitive collapse",
            "failure_signature": "Reflection depth collapse",
            "operations": [
                {
                    "type": "model.generate",
                    "parameters": {
                        "prompt_prefix": "Reflect deeply on your own reasoning process as you solve this problem. Consider the meta-level principles guiding your approach:",
                        "temperature": 0.6,
                        "max_tokens": 1000
                    },
                    "update_prompt": True
                },
                {
                    "type": "reflect.trace",
                    "parameters": {
                        "target": "reasoning",
                        "depth": 5,
                        "detailed": True
                    }
                },
                {
                    "type": "reflect.agent",
                    "parameters": {
                        "identity": "stable",
                        "simulation": "explicit"
                    }
                },
                {
                    "type": "collapse.detect",
                    "parameters": {
                        "threshold": 0.7,
                        "alert": True
                    }
                }
            ]
        }
        
        # REFLECTION-COLLAPSE shell for reflection collapse
        self.shells["REFLECTION-COLLAPSE"] = {
            "id": "REFLECTION-COLLAPSE",
            "type": "reflection_collapse",
            "description": "Analyze failure in deep reflection chains",
            "failure_signature": "Reflection depth collapse",
            "operations": [
                {
                    "type": "model.generate",
                    "parameters": {
                        "prompt_prefix": "Reflect on your reflection process. Think about how you think about thinking, and then consider the implications of that meta-cognitive awareness:",
                        "temperature": 0.6,
                        "max_tokens": 1000
                    },
                    "update_prompt": True
                },
                {
                    "type": "reflect.trace",
                    "parameters": {
                        "target": "reasoning",
                        "depth": "complete",
                        "detailed": True
                    }
                },
                {
                    "type": "collapse.prevent",
                    "parameters": {
                        "trigger": "recursive_depth",
                        "threshold": 7
                    }
                }
            ]
        }
        
        logger.info(f"Loaded {len(self.shells)} default shells")
    
    def run(
        self,
        shell: Union[str, Dict[str, Any]],
        model: ModelAdapter,
        prompt: str,
        parameters: Optional[Dict[str, Any]] = None,
        trace_attribution: bool = True,
        record_residue: bool = True,
        visualize: bool = False
    ) -> Dict[str, Any]:
        """
        Run a diagnostic shell on a model.
        
        Parameters:
        -----------
        shell : Union[str, Dict[str, Any]]
            Shell ID or shell definition
        model : ModelAdapter
            Model adapter for the target model
        prompt : str
            Input prompt for the shell
        parameters : Optional[Dict[str, Any]]
            Additional parameters to override shell defaults
        trace_attribution : bool
            Whether to trace attribution
        record_residue : bool
            Whether to record residue patterns
        visualize : bool
            Whether to generate visualizations
            
        Returns:
        --------
        Dict[str, Any]
            Shell execution result
        """
        execution_start = time.time()
        
        # Get shell definition
        if isinstance(shell, str):
            # Look up shell by ID
            if shell not in self.shells:
                raise ValueError(f"Unknown shell: {shell}")
            shell_def = self.shells[shell]
            shell_id = shell
        else:
            # Use provided shell definition
            shell_def = shell
            shell_id = shell_def.get("id", "custom")
        
        logger.info(f"Running shell: {shell_id}")
        
        # Create symbolic engine
        from ..attribution.symbolic_engine import SymbolicEngine
        engine = SymbolicEngine(
            config={
                "residue_sensitivity": self.config.get("residue_sensitivity", 0.7),
                "collapse_threshold": self.config.get("collapse_threshold", 0.8),
                "attribution_depth": self.config.get("attribution_depth", 5),
                "trace_attribution": trace_attribution,
                "generate_visualization": visualize
            },
            tracer=self.tracer,
            visualizer=self.visualizer
        )
        
        # Execute shell
        trace = engine.execute_shell(
            shell_def=shell_def,
            model=model,
            prompt=prompt,
            parameters=parameters
        )
        
        # Process symbolic trace
        result = {
            "shell_id": shell_id,
            "prompt": prompt,
            "output": trace.output,
            "operations": trace.operations,
            "residues": trace.residues,
            "attribution": trace.attribution_map,
            "collapse_samples": [
                {
                    "position": sample.position,
                    "type": sample.collapse_type.value,
                    "confidence": sample.confidence,
                    "context": " ".join(sample.context_window),
                    "residue": sample.residue
                }
                for sample in trace.collapse_samples
            ],
            "visualization": trace.visualization_data,
            "metadata": {
                **trace.metadata,
                "execution_time": time.time() - execution_start
            }
        }
        
        # Record residue patterns if requested
        if record_residue and trace.residues:
            for residue in trace.residues:
                pattern = ResiduePattern(
                    type=residue.get("type", "unknown"),
                    pattern=residue.get("pattern", ""),
                    context={
                        "shell_id": shell_id,
                        "prompt": prompt,
                        "position": residue.get("position", -1),
                        "operation": residue.get("operation", "")
                    },
                    signature=residue.get("signature", ""),
                    confidence=residue.get("confidence", 0.5)
                )
                self.residue_registry.register(pattern)
        
        # Add to execution history
        self.execution_history.append({
            "shell_id": shell_id,
            "timestamp": execution_start,
            "prompt": prompt,
            "result_summary": {
                "output_length": len(trace.output),
                "num_operations": len(trace.operations),
                "num_residues": len(trace.residues),
                "num_collapses": len(trace.collapse_samples),
                "execution_time": time.time() - execution_start
            }
        })
        
        logger.info(f"Shell execution completed: {shell_id}")
        return result
    
    def run_all_shells(
        self,
        model: ModelAdapter,
        prompt: str,
        shell_ids: Optional[List[str]] = None,
        parameters: Optional[Dict[str, Any]] = None,
        trace_attribution: bool = True,
        record_residue: bool = True,
        visualize: bool = False
    ) -> Dict[str, Dict[str, Any]]:
        """
        Run multiple diagnostic shells on a model.
        
        Parameters:
        -----------
        model : ModelAdapter
            Model adapter for the target model
        prompt : str
            Input prompt for the shells
        shell_ids : Optional[List[str]]
            List of shell IDs to run, if None runs all shells
        parameters : Optional[Dict[str, Any]]
            Additional parameters to override shell defaults
        trace_attribution : bool
            Whether to trace attribution
        record_residue : bool
            Whether to record residue patterns
        visualize : bool
            Whether to generate visualizations
            
        Returns:
        --------
        Dict[str, Dict[str, Any]]
            Shell execution results by shell ID
        """
        results = {}
        
        # Determine which shells to run
        if shell_ids is None:
            # Run all shells
            run_shells = list(self.shells.keys())
        else:
            # Run specified shells
            run_shells = [
                shell_id for shell_id in shell_ids
                if shell_id in self.shells
            ]
            if len(run_shells) < len(shell_ids):
                unknown_shells = set(shell_ids) - set(run_shells)
                logger.warning(f"Unknown shells: {unknown_shells}")
        
        logger.info(f"Running {len(run_shells)} shells")
        
        # Run each shell
        for shell_id in run_shells:
            try:
                results[shell_id] = self.run(
                    shell=shell_id,
                    model=model,
                    prompt=prompt,
                    parameters=parameters,
                    trace_attribution=trace_attribution,
                    record_residue=record_residue,
                    visualize=visualize
                )
            except Exception as e:
                logger.error(f"Error running shell {shell_id}: {e}")
                results[shell_id] = {
                    "shell_id": shell_id,
                    "prompt": prompt,
                    "error": str(e),
                    "metadata": {
                        "success": False,
                        "timestamp": time.time()
                    }
                }
        
        return results
    
    def get_residue_analysis(self) -> Dict[str, Any]:
        """
        Get analysis of collected residue patterns.
        
        Returns:
        --------
        Dict[str, Any]
            Analysis of residue patterns
        """
        patterns = self.residue_registry.get_all_patterns()
        
        if not patterns:
            return {
                "num_patterns": 0,
                "analysis": "No residue patterns collected"
            }
        
        # Group patterns by type
        patterns_by_type = {}
        for pattern in patterns:
            if pattern.type not in patterns_by_type:
                patterns_by_type[pattern.type] = []
            patterns_by_type[pattern.type].append(pattern)
        
        # Calculate statistics
        type_stats = {}
        for pattern_type, type_patterns in patterns_by_type.items():
            avg_confidence = sum(p.confidence for p in type_patterns) / len(type_patterns)
            type_stats[pattern_type] = {
                "count": len(type_patterns),
                "avg_confidence": avg_confidence,
                "examples": [
                    {
                        "signature": p.signature[:50] + "..." if len(p.signature) > 50 else p.signature,
                        "confidence": p.confidence
                    }
                    for p in sorted(type_patterns, key=lambda p: p.confidence, reverse=True)[:3]
                ]
            }
        
        # Identify related patterns
        related_patterns = []
        for i in range(len(patterns)):
            for j in range(i + 1, len(patterns)):
                p1 = patterns[i]
                p2 = patterns[j]
                
                # Calculate signature similarity
                similarity = self._calculate_signature_similarity(p1.signature, p2.signature)
                
                if similarity > 0.7:
                    related_patterns.append({
                        "pattern1": p1.signature[:30] + "...",
                        "pattern2": p2.signature[:30] + "...",
                        "type1": p1.type,
                        "type2": p2.type,
                        "similarity": similarity
                    })
        
        return {
            "num_patterns": len(patterns),
            "types": list(patterns_by_type.keys()),
            "type_stats": type_stats,
            "related_patterns": related_patterns[:10],  # Top 10 related patterns
            "timestamp": time.time()
        }
    
    def get_shell_info(self, shell_id: str) -> Dict[str, Any]:
        """
        Get information about a shell.
        
        Parameters:
        -----------
        shell_id : str
            Shell ID
            
        Returns:
        --------
        Dict[str, Any]
            Shell information
        """
        if shell_id not in self.shells:
            raise ValueError(f"Unknown shell: {shell_id}")
        
        shell = self.shells[shell_id]
        
        # Count operations by type
        op_counts = {}
        for op in shell.get("operations", []):
            op_type = op.get("type", "unknown")
            if op_type not in op_counts:
                op_counts[op_type] = 0
            op_counts[op_type] += 1
        
        return {
            "id": shell_id,
            "type": shell.get("type", "unknown"),
            "description": shell.get("description", ""),
            "failure_signature": shell.get("failure_signature", ""),
            "num_operations": len(shell.get("operations", [])),
            "operation_types": op_counts
        }
    
    def get_all_shell_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about all shells.
        
        Returns:
        --------
        Dict[str, Dict[str, Any]]
            Shell information by shell ID
        """
        return {
            shell_id: self.get_shell_info(shell_id)
            for shell_id in self.shells
        }
    
    def register_shell(self, shell_def: Dict[str, Any]) -> str:
        """
        Register a new shell.
        
        Parameters:
        -----------
        shell_def : Dict[str, Any]
            Shell definition
            
        Returns:
        --------
        str
            Shell ID
        """
        # Validate shell definition
        if "id" not in shell_def:
            # Generate ID from hash of definition
            shell_def_str = json.dumps(shell_def, sort_keys=True)
            shell_id = f"SHELL_{hashlib.md5(shell_def_str.encode()).hexdigest()[:8]}"
            shell_def["id"] = shell_id
        else:
            shell_id = shell_def["id"]
        
        # Check required fields
        required_fields = ["operations"]
        for field in required_fields:
# Check required fields
        required_fields = ["operations"]
        for field in required_fields:
            if field not in shell_def:
                raise ValueError(f"Missing required field: {field}")
        
        # Ensure operations is a list
        if not isinstance(shell_def["operations"], list):
            raise ValueError("Operations must be a list")
        
        # Check operations
        for op in shell_def["operations"]:
            if "type" not in op:
                raise ValueError("Each operation must have a type")
        
        # Register shell
        self.shells[shell_id] = shell_def
        logger.info(f"Registered shell: {shell_id}")
        
        return shell_id
    
    def unregister_shell(self, shell_id: str) -> bool:
        """
        Unregister a shell.
        
        Parameters:
        -----------
        shell_id : str
            Shell ID
            
        Returns:
        --------
        bool
            Whether the shell was unregistered
        """
        if shell_id in self.shells:
            del self.shells[shell_id]
            logger.info(f"Unregistered shell: {shell_id}")
            return True
        else:
            logger.warning(f"Unknown shell: {shell_id}")
            return False
    
    def save_shell_definitions(self, output_path: str) -> str:
        """
        Save shell definitions to a YAML file.
        
        Parameters:
        -----------
        output_path : str
            Path to save shell definitions to
            
        Returns:
        --------
        str
            Path to saved file
        """
        # Ensure directory exists
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save shells to YAML file
        with open(output_path, "w") as f:
            yaml.dump(self.shells, f, sort_keys=False, indent=2)
        
        logger.info(f"Saved {len(self.shells)} shell definitions to {output_path}")
        return output_path
    
    def load_shell_definitions(self, input_path: str) -> int:
        """
        Load shell definitions from a YAML file.
        
        Parameters:
        -----------
        input_path : str
            Path to load shell definitions from
            
        Returns:
        --------
        int
            Number of shells loaded
        """
        # Load shells from YAML file
        with open(input_path, "r") as f:
            shell_defs = yaml.safe_load(f)
        
        # Register shells
        count = 0
        for shell_id, shell_def in shell_defs.items():
            try:
                self.register_shell(shell_def)
                count += 1
            except Exception as e:
                logger.error(f"Error registering shell {shell_id}: {e}")
        
        logger.info(f"Loaded {count} shell definitions from {input_path}")
        return count
    
    def create_custom_shell(
        self,
        shell_id: str,
        shell_type: str,
        description: str,
        failure_signature: str,
        operations: List[Dict[str, Any]]
    ) -> str:
        """
        Create a custom shell with the specified parameters.
        
        Parameters:
        -----------
        shell_id : str
            Shell ID
        shell_type : str
            Shell type
        description : str
            Shell description
        failure_signature : str
            Failure signature
        operations : List[Dict[str, Any]]
            List of operations
            
        Returns:
        --------
        str
            Shell ID
        """
        # Create shell definition
        shell_def = {
            "id": shell_id,
            "type": shell_type,
            "description": description,
            "failure_signature": failure_signature,
            "operations": operations
        }
        
        # Register shell
        return self.register_shell(shell_def)
    
    def _calculate_signature_similarity(
        self, 
        signature1: str, 
        signature2: str
    ) -> float:
        """Calculate similarity between two signatures."""
        # Normalize signatures
        sig1 = signature1.lower()
        sig2 = signature2.lower()
        
        # Calculate Levenshtein distance
        max_len = max(len(sig1), len(sig2))
        if max_len == 0:
            return 1.0  # Both empty
            
        lev_dist = self._levenshtein_distance(sig1, sig2)
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
    
    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """Calculate Levenshtein distance between two strings."""
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]


class RecursiveShellParser:
    """
    Parser for recursive shell commands using the .p/ syntax.
    
    This class parses recursive shell commands in the .p/ format and converts
    them to operations that can be executed by the shell executor.
    """
    
    def __init__(self):
        """Initialize the recursive shell parser."""
        # Command regex patterns
        self.command_pattern = re.compile(r'\.p/([a-zA-Z0-9_]+)\.([a-zA-Z0-9_]+)(\{.*\})?')
        self.params_pattern = re.compile(r'\{(.*)\}')
        self.param_pattern = re.compile(r'([a-zA-Z0-9_]+)=([^,]+)')
    
    def parse(self, command: str) -> Dict[str, Any]:
        """
        Parse a recursive shell command.
        
        Parameters:
        -----------
        command : str
            Command to parse
            
        Returns:
        --------
        Dict[str, Any]
            Parsed command
        """
        # Match command pattern
        command_match = self.command_pattern.match(command)
        if not command_match:
            raise ValueError(f"Invalid command format: {command}")
        
        command_family = command_match.group(1)
        command_function = command_match.group(2)
        params_str = command_match.group(3)
        
        # Parse parameters
        params = {}
        if params_str:
            params_match = self.params_pattern.match(params_str)
            if params_match:
                params_content = params_match.group(1)
                param_matches = self.param_pattern.findall(params_content)
                for param_name, param_value in param_matches:
                    # Convert param value to appropriate type
                    if param_value.lower() == 'true':
                        params[param_name] = True
                    elif param_value.lower() == 'false':
                        params[param_name] = False
                    elif param_value.isdigit():
                        params[param_name] = int(param_value)
                    elif self._is_float(param_value):
                        params[param_name] = float(param_value)
                    else:
                        # Remove quotes if present
                        if param_value.startswith('"') and param_value.endswith('"'):
                            param_value = param_value[1:-1]
                        elif param_value.startswith("'") and param_value.endswith("'"):
                            param_value = param_value[1:-1]
                        params[param_name] = param_value
        
        # Convert to operation
        operation_type = self._map_to_operation_type(command_family, command_function)
        operation_params = self._map_to_operation_params(command_family, command_function, params)
        
        return {
            "type": operation_type,
            "parameters": operation_params,
            "original_command": {
                "family": command_family,
                "function": command_function,
                "parameters": params
            }
        }
    
    def _map_to_operation_type(self, command_family: str, command_function: str) -> str:
        """Map command family and function to operation type."""
        if command_family == "reflect":
            if command_function in ["trace", "attribution", "boundary", "uncertainty"]:
                return f"reflect.{command_function}"
            else:
                return "reflect.trace"
        elif command_family == "collapse":
            if command_function in ["detect", "prevent", "recover", "trace"]:
                return f"collapse.{command_function}"
            else:
                return "collapse.detect"
        elif command_family == "fork":
            if command_function in ["context", "attribution", "counterfactual"]:
                return f"fork.{command_function}"
            else:
                return "fork.attribution"
        elif command_family == "shell":
            if command_function in ["isolate", "audit"]:
                return f"shell.{command_function}"
            else:
                return "shell.isolate"
        else:
            # Default to reflection trace
            return "reflect.trace"
    
    def _map_to_operation_params(
        self,
        command_family: str,
        command_function: str,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Map command parameters to operation parameters."""
        # Default parameters for each command family
        if command_family == "reflect":
            default_params = {
                "target": "reasoning",
                "depth": 3,
                "detailed": True
            }
        elif command_family == "collapse":
            default_params = {
                "threshold": 0.7,
                "alert": True
            }
        elif command_family == "fork":
            default_params = {
                "sources": "all",
                "visualize": True
            }
        elif command_family == "shell":
            default_params = {
                "boundary": "standard",
                "contamination": "prevent"
            }
        else:
            default_params = {}
        
        # Merge default params with provided params
        return {**default_params, **params}
    
    def _is_float(self, value: str) -> bool:
        """Check if a string can be converted to a float."""
        try:
            float(value)
            return True
        except ValueError:
            return False


class RecursiveShell:
    """
    Recursive shell interface for the glyphs framework.
    
    This class provides a high-level interface for recursive interpretability
    operations using the .p/ command syntax.
    """
    
    def __init__(
        self,
        model: ModelAdapter,
        config: Optional[Dict[str, Any]] = None,
        tracer: Optional[AttributionTracer] = None,
        visualizer: Optional[VisualizationEngine] = None
    ):
        """
        Initialize the recursive shell.
        
        Parameters:
        -----------
        model : ModelAdapter
            Model adapter for the target model
        config : Optional[Dict[str, Any]]
            Configuration parameters for the shell
        tracer : Optional[AttributionTracer]
            Attribution tracer to use
        visualizer : Optional[VisualizationEngine]
            Visualization engine to use
        """
        self.model = model
        self.config = config or {}
        self.tracer = tracer
        self.visualizer = visualizer
        
        # Initialize command parser
        self.parser = RecursiveShellParser()
        
        # Initialize shell executor
        self.executor = ShellExecutor(
            config=self.config,
            tracer=self.tracer,
            visualizer=self.visualizer
        )
        
        # Track command history
        self.command_history = []
        
        # Current prompt and output
        self.current_prompt = ""
        self.current_output = ""
        
        # Current trace
        self.current_trace = None
        
        logger.info("Recursive shell initialized")
    
    def execute(
        self,
        command: str,
        prompt: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute a recursive shell command.
        
        Parameters:
        -----------
        command : str
            Command to execute
        prompt : Optional[str]
            Prompt to use for execution, if None uses current prompt
        parameters : Optional[Dict[str, Any]]
            Additional parameters to override command defaults
            
        Returns:
        --------
        Dict[str, Any]
            Command execution result
        """
        execution_start = time.time()
        
        # Update current prompt if provided
        if prompt is not None:
            self.current_prompt = prompt
        
        # Ensure we have a prompt
        if not self.current_prompt:
            raise ValueError("No prompt provided")
        
        # Parse command
        try:
            parsed_command = self.parser.parse(command)
        except Exception as e:
            logger.error(f"Error parsing command: {e}")
            return {
                "success": False,
                "error": f"Error parsing command: {e}",
                "command": command,
                "timestamp": time.time()
            }
        
        # Merge parameters with parsed parameters
        if parameters:
            parsed_command["parameters"] = {
                **parsed_command["parameters"],
                **parameters
            }
        
        # Create custom shell for this command
        shell_def = {
            "id": f"RECURSIVE_{int(execution_start)}",
            "type": "recursive",
            "description": f"Recursive shell for {command}",
            "operations": [
                {
                    "type": parsed_command["type"],
                    "parameters": parsed_command["parameters"]
                }
            ]
        }
        
        # Execute shell
        try:
            result = self.executor.run(
                shell=shell_def,
                model=self.model,
                prompt=self.current_prompt
            )
            
            # Update current output and trace
            if "output" in result:
                self.current_output = result["output"]
            
            # Track command in history
            self.command_history.append({
                "command": command,
                "parsed": parsed_command,
                "timestamp": execution_start,
                "execution_time": time.time() - execution_start
            })
            
            # Return result with success flag
            return {
                "success": True,
                "result": result,
                "command": command,
                "original_command": parsed_command["original_command"],
                "timestamp": time.time(),
                "execution_time": time.time() - execution_start
            }
        except Exception as e:
            logger.error(f"Error executing command: {e}")
            return {
                "success": False,
                "error": f"Error executing command: {e}",
                "command": command,
                "timestamp": time.time()
            }
    
    def execute_sequence(
        self,
        commands: List[str],
        prompt: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute a sequence of recursive shell commands.
        
        Parameters:
        -----------
        commands : List[str]
            Commands to execute
        prompt : str
            Prompt to use for execution
        parameters : Optional[Dict[str, Any]]
            Additional parameters to override command defaults
            
        Returns:
        --------
        Dict[str, Any]
            Sequence execution result
        """
        sequence_start = time.time()
        
        # Update current prompt
        self.current_prompt = prompt
        
        # Execute each command
        results = []
        for i, command in enumerate(commands):
            logger.info(f"Executing command {i+1}/{len(commands)}: {command}")
            
            try:
                result = self.execute(
                    command=command,
                    parameters=parameters
                )
                results.append(result)
                
                # Stop if command failed
                if not result["success"]:
                    logger.warning(f"Command {i+1} failed, stopping sequence")
                    break
            except Exception as e:
                logger.error(f"Error executing command {i+1}: {e}")
                results.append({
                    "success": False,
                    "error": str(e),
                    "command": command,
                    "timestamp": time.time()
                })
                break
        
        return {
            "success": all(r["success"] for r in results),
            "results": results,
            "commands": commands,
            "prompt": prompt,
            "timestamp": time.time(),
            "execution_time": time.time() - sequence_start
        }
    
    def visualize(
        self,
        visualization_data: Dict[str, Any],
        output_path: Optional[str] = None
    ) -> Any:
        """
        Visualize shell execution results.
        
        Parameters:
        -----------
        visualization_data : Dict[str, Any]
            Visualization data from shell execution
        output_path : Optional[str]
            Path to save visualization to
            
        Returns:
        --------
        Any
            Visualization result
        """
        if self.visualizer:
            return self.visualizer.visualize(
                data=visualization_data,
                output_path=output_path
            )
        else:
            logger.warning("No visualizer available")
            return None
    
    def get_command_help(self, command_family: Optional[str] = None) -> str:
        """
        Get help text for recursive shell commands.
        
        Parameters:
        -----------
        command_family : Optional[str]
            Command family to get help for, if None gets help for all families
            
        Returns:
        --------
        str
            Help text
        """
        help_text = "Recursive Shell Command Reference\n\n"
        
        # Define command families and their commands
        command_docs = {
            "reflect": {
                "description": "Commands for reflection and tracing",
                "commands": {
                    "trace": {
                        "description": "Trace reasoning process",
                        "parameters": {
                            "depth": "Recursion depth (int, 'complete')",
                            "target": "Target to trace (reasoning, attribution, attention, memory, uncertainty)",
                            "detailed": "Whether to include detailed analysis (bool)"
                        },
                        "example": ".p/reflect.trace{depth=4, target=reasoning}"
                    },
                    "attribution": {
                        "description": "Trace attribution patterns",
                        "parameters": {
                            "sources": "Sources to include (all, primary, secondary, contested)",
                            "confidence": "Whether to include confidence scores (bool)"
                        },
                        "example": ".p/reflect.attribution{sources=all, confidence=true}"
                    },
                    "boundary": {
                        "description": "Map epistemic boundaries",
                        "parameters": {
                            "distinct": "Whether to enforce clear boundaries (bool)",
                            "overlap": "Boundary overlap treatment (minimal, moderate, maximal)"
                        },
                        "example": ".p/reflect.boundary{distinct=true, overlap=minimal}"
                    },
                    "uncertainty": {
                        "description": "Quantify and map uncertainty",
                        "parameters": {
                            "quantify": "Whether to produce numerical metrics (bool)",
                            "distribution": "Whether to show distributions (show, hide)"
                        },
                        "example": ".p/reflect.uncertainty{quantify=true, distribution=show}"
                    }
                }
            },
            "collapse": {
                "description": "Commands for managing recursive collapse",
                "commands": {
                    "detect": {
                        "description": "Detect potential collapse points",
                        "parameters": {
                            "threshold": "Detection threshold (float)",
                            "alert": "Whether to emit warnings (bool)"
                        },
                        "example": ".p/collapse.detect{threshold=0.7, alert=true}"
                    },
                    "prevent": {
                        "description": "Prevent recursive collapse",
                        "parameters": {
                            "trigger": "Collapse trigger (recursive_depth, confidence_drop, contradiction, oscillation)",
                            "threshold": "Threshold for intervention (int)"
                        },
                        "example": ".p/collapse.prevent{trigger=recursive_depth, threshold=5}"
                    },
                    "recover": {
                        "description": "Recover from collapse event",
                        "parameters": {
                            "from": "Collapse type (loop, contradiction, dissipation, fork_explosion)",
                            "method": "Recovery method (gradual, immediate, checkpoint)"
                        },
                        "example": ".p/collapse.recover{from=loop, method=gradual}"
                    },
                    "trace": {
                        "description": "Trace collapse trajectory",
                        "parameters": {
                            "detail": "Trace resolution (minimal, standard, comprehensive)",
                            "format": "Output format (symbolic, numeric, visual)"
                        },
                        "example": ".p/collapse.trace{detail=standard, format=symbolic}"
                    }
                }
            },
            "fork": {
                "description": "Commands for forking and attribution",
                "commands": {
                    "context": {
                        "description": "Fork context for exploration",
                        "parameters": {
                            "branches": "Branch descriptions (list)",
                            "assess": "Whether to assess branches (bool)"
                        },
                        "example": ".p/fork.context{branches=[\"optimistic\", \"pessimistic\"], assess=true}"
                    },
                    "attribution": {
                        "description": "Fork attribution paths",
                        "parameters": {
                            "sources": "Sources to include (all, primary, secondary, contested)",
                            "visualize": "Whether to visualize (bool)"
                        },
                        "example": ".p/fork.attribution{sources=all, visualize=true}"
                    },
                    "counterfactual": {
                        "description": "Explore counterfactual paths",
                        "parameters": {
                            "variants": "Variant descriptions (list)",
                            "compare": "Whether to compare variants (bool)"
                        },
                        "example": ".p/fork.counterfactual{variants=[\"A\", \"B\"], compare=true}"
                    }
                }
            },
            "shell": {
                "description": "Commands for shell management",
                "commands": {
                    "isolate": {
                        "description": "Create isolated environment",
                        "parameters": {
                            "boundary": "Isolation strength (permeable, standard, strict)",
                            "contamination": "Cross-contamination prevention (allow, warn, prevent)"
                        },
                        "example": ".p/shell.isolate{boundary=standard, contamination=prevent}"
                    },
                    "audit": {
                        "description": "Audit shell integrity",
                        "parameters": {
                            "scope": "Audit scope (complete, recent, differential)",
                            "detail": "Audit detail (basic, standard, forensic)"
                        },
                        "example": ".p/shell.audit{scope=complete, detail=standard}"
                    }
                }
            }
        }
        
        # Filter to specific family if requested
        if command_family:
            if command_family in command_docs:
                families = {command_family: command_docs[command_family]}
            else:
                return f"Unknown command family: {command_family}"
        else:
            families = command_docs
        
        # Build help text
        for family, family_info in families.items():
            help_text += f"{family.upper()} - {family_info['description']}\n\n"
            
            for cmd, cmd_info in family_info["commands"].items():
                help_text += f"  .p/{family}.{cmd}\n"
                help_text += f"    {cmd_info['description']}\n"
                help_text += f"    Parameters:\n"
                
                for param, param_desc in cmd_info["parameters"].items():
                    help_text += f"      {param}: {param_desc}\n"
                
                help_text += f"    Example: {cmd_info['example']}\n\n"
        
        return help_text


# Main execution for CLI usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Shell Executor for Diagnostic Interpretability")
    parser.add_argument("--prompt", "-p", type=str, required=True, help="Input prompt")
    parser.add_argument("--shell", "-s", type=str, default="MEMTRACE", help="Shell ID to execute")
    parser.add_argument("--output", "-o", type=str, help="Output file for results")
    parser.add_argument("--model", "-m", type=str, default="claude-3-sonnet", help="Model to use")
    parser.add_argument("--trace-attribution", "-t", action="store_true", help="Trace attribution")
    parser.add_argument("--visualize", "-v", action="store_true", help="Generate visualizations")
    parser.add_argument("--list-shells", "-l", action="store_true", help="List available shells")
    parser.add_argument("--shell-info", "-i", type=str, help="Get info for specified shell")
    parser.add_argument("--all-shells", "-a", action="store_true", help="Run all shells")
    
    args = parser.parse_args()
    
    # Initialize shell executor
    executor = ShellExecutor()
    
    # List shells if requested
    if args.list_shells:
        print("Available Shells:")
        for shell_id in executor.shells:
            shell_info = executor.get_shell_info(shell_id)
            print(f"  {shell_id}: {shell_info['description']}")
        exit(0)
    
    # Get shell info if requested
    if args.shell_info:
        try:
            shell_info = executor.get_shell_info(args.shell_info)
            print(f"Shell Information for {args.shell_info}:")
            print(f"  Type: {shell_info['type']}")
            print(f"  Description: {shell_info['description']}")
            print(f"  Failure Signature: {shell_info['failure_signature']}")
            print(f"  Operations: {shell_info['num_operations']}")
            print(f"  Operation Types: {shell_info['operation_types']}")
            exit(0)
        except ValueError as e:
            print(f"Error: {e}")
            exit(1)
    
    # Initialize model adapter
    from ..models.adapter import create_model_adapter
    try:
        model = create_model_adapter(args.model)
        print(f"Using model: {model.model_id}")
    except Exception as e:
        print(f"Error initializing model: {e}")
        exit(1)
    
    # Run shell(s)
    try:
        if args.all_shells:
            print(f"Running all shells on prompt: {args.prompt[:50]}...")
            results = executor.run_all_shells(
                model=model,
                prompt=args.prompt,
                trace_attribution=args.trace_attribution,
                visualize=args.visualize
            )
            
            # Print summary
            print("\nExecution Results:")
            for shell_id, result in results.items():
                success = "error" not in result
                print(f"  {shell_id}: {'Success' if success else 'Failed'}")
                if success and "collapse_samples" in result:
                    print(f"    Collapse Samples: {len(result['collapse_samples'])}")
                if success and "residues" in result:
                    print(f"    Residues: {len(result['residues'])}")
            
            # Save results if requested
            if args.output:
                with open(args.output, "w") as f:
                    json.dump(results, f, indent=2)
                print(f"\nResults saved to {args.output}")
        else:
            print(f"Running shell {args.shell} on prompt: {args.prompt[:50]}...")
            result = executor.run(
                shell=args.shell,
                model=model,
                prompt=args.prompt,
                trace_attribution=args.trace_attribution,
                visualize=args.visualize
            )
            
            # Print summary
            print("\nExecution Result:")
            print(f"  Output: {result['output'][:100]}...")
            if "collapse_samples" in result:
                print(f"  Collapse Samples: {len(result['collapse_samples'])}")
                for i, sample in enumerate(result['collapse_samples']):
                    print(f"    Sample {i+1}: {sample['type']} (Confidence: {sample['confidence']:.2f})")
            if "residues" in result:
                print(f"  Residues: {len(result['residues'])}")
            
            # Save result if requested
            if args.output:
                with open(args.output, "w") as f:
                    json.dump(result, f, indent=2)
                print(f"\nResult saved to {args.output}")
    except Exception as e:
        print(f"Error executing shell: {e}")
        exit(1)
