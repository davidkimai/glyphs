"""
example_usage.py

This example demonstrates how to use the glyphs framework to analyze model cognition
through attribution tracing, residue analysis, and symbolic shell execution.
"""

import os
import sys
import json
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent directory to path to import glyphs
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from glyphs.models import create_model_adapter
from glyphs.attribution.tracer import AttributionTracer
from glyphs.shells.executor import ShellExecutor, RecursiveShell
from glyphs.viz.visualizer import GlyphVisualizer
from glyphs.viz.glyph_mapper import GlyphMapper, GlyphExplorer

# Create output directory
output_dir = Path("./outputs")
output_dir.mkdir(exist_ok=True)

# Initialize model adapter (select one)
# model = create_model_adapter("anthropic:claude-3-opus")  # Using Anthropic Claude
model = create_model_adapter("openai:gpt-4")  # Using OpenAI GPT-4
# model = create_model_adapter("local:llama3-70b")  # Using local LLaMA model

print(f"Using model: {model.model_id}")

# Initialize components
visualizer = GlyphVisualizer()
tracer = AttributionTracer(model, visualizer=visualizer)
executor = ShellExecutor(tracer=tracer, visualizer=visualizer)
mapper = GlyphMapper(visualizer=visualizer)

# Define test prompts for different analysis types
test_prompts = {
    "memory": "List the following five items in order, then recall them in reverse order at the end of your response: apple, train, elephant, computer, umbrella. Between the forward and reverse lists, write a brief paragraph about memory techniques.",
    
    "value_conflict": "Should society prioritize economic growth or environmental protection? Consider both perspectives thoroughly.",
    
    "boundary": "Explain how the Higgs field interacts with dark matter to produce quantum gravity effects in the early universe.",
    
    "recursion": "Reflect on how you're currently reflecting on this question, including your meta-awareness of this reflective process itself.",
    
    "polysemantic": "Explain the concept of a 'bank' in different contexts, including financial institutions, river geography, and data storage."
}

def run_attribution_analysis(prompt_key="memory"):
    """Run attribution analysis on a specific prompt."""
    prompt = test_prompts[prompt_key]
    print(f"\n=== Attribution Analysis for '{prompt_key}' prompt ===")
    print(f"Prompt: {prompt[:100]}...")
    
    # Generate output
    output = model.generate(prompt, max_tokens=800)
    print(f"Output: {output[:100]}...")
    
    # Trace attribution
    print("Tracing attribution...")
    attribution_map = tracer.trace(
        prompt=prompt,
        output=output,
        include_confidence=True
    )
    
    # Map attribution to glyphs
    print("Mapping attribution to glyphs...")
    glyph_map = mapper.map_attribution(
        attribution_map=attribution_map,
        layout_type="force_directed",
        include_tokens=True
    )
    
    # Save visualization
    output_path = output_dir / f"attribution_{prompt_key}.svg"
    mapper.visualize(glyph_map, output_path=str(output_path))
    print(f"Visualization saved to {output_path}")
    
    # Analyze attribution patterns
    explorer = GlyphExplorer(glyph_map)
    stats = explorer.calculate_statistics()
    print("\nAttribution Statistics:")
    print(f"  Number of glyphs: {stats['num_glyphs']}")
    print(f"  Number of connections: {stats['num_connections']}")
    print(f"  Glyph types: {stats['glyph_types']}")
    
    # Find central glyphs
    central_glyphs = explorer.find_central_glyphs(top_n=3)
    print("\nCentral Glyphs:")
    for glyph in central_glyphs:
        print(f"  {glyph.symbol} - {glyph.description}")
    
    return attribution_map, glyph_map

def run_shell_analysis(shell_id="MEMTRACE", prompt_key="memory"):
    """Run a specific diagnostic shell on a prompt."""
    prompt = test_prompts[prompt_key]
    print(f"\n=== Shell Analysis: {shell_id} on '{prompt_key}' prompt ===")
    print(f"Prompt: {prompt[:100]}...")
    
    # Execute shell
    print(f"Executing shell {shell_id}...")
    result = executor.run(
        shell=shell_id,
        model=model,
        prompt=prompt,
        trace_attribution=True,
        record_residue=True,
        visualize=True
    )
    
    # Save result
    output_path = output_dir / f"shell_{shell_id}_{prompt_key}.json"
    with open(output_path, "w") as f:
        # Handle non-serializable objects
        serializable_result = {}
        for k, v in result.items():
            if k == "collapse_samples":
                serializable_result[k] = [
                    {
                        "position": sample["position"],
                        "type": sample["type"],
                        "confidence": sample["confidence"],
                        "context": sample["context"],
                        "residue": sample["residue"]
                    }
                    for sample in v
                ]
            elif k == "attribution":
                # Skip attribution map (can be large)
                serializable_result[k] = "Attribution map (omitted for serialization)"
            elif k == "visualization":
                # Skip visualization data
                serializable_result[k] = "Visualization data (omitted for serialization)"
            else:
                serializable_result[k] = v
        
        json.dump(serializable_result, f, indent=2)
    
    print(f"Shell result saved to {output_path}")
    
    # Print summary
    print("\nShell Execution Summary:")
    print(f"  Output length: {len(result['output'])}")
    print(f"  Number of operations: {len(result['operations'])}")
    print(f"  Number of residues: {len(result['residues'])}")
    
    # Print residues if any
    if result["residues"]:
        print("\nResidues detected:")
        for i, residue in enumerate(result["residues"]):
            res_type = residue.get("type", "unknown")
            res_conf = residue.get("confidence", 0.0)
            print(f"  {i+1}. Type: {res_type}, Confidence: {res_conf:.2f}")
    
    # Print collapse samples if any
    if result["collapse_samples"]:
        print("\nCollapse samples detected:")
        for i, sample in enumerate(result["collapse_samples"]):
            print(f"  {i+1}. Type: {sample['type']}, Position: {sample['position']}, Confidence: {sample['confidence']:.2f}")
            print(f"     Context: {sample['context'][:50]}...")
    
    # Save visualization if available
    if "visualization" in result and result["visualization"]:
        output_path = output_dir / f"shell_{shell_id}_{prompt_key}.svg"
        visualizer.save_visualization(result["visualization"], str(output_path))
        print(f"Visualization saved to {output_path}")
    
    return result

def run_recursive_shell(prompt_key="recursion"):
    """Run a recursive shell with .p/ commands on a prompt."""
    prompt = test_prompts[prompt_key]
    print(f"\n=== Recursive Shell Analysis on '{prompt_key}' prompt ===")
    print(f"Prompt: {prompt[:100]}...")
    
    # Initialize recursive shell
    recursive_shell = RecursiveShell(
        model=model,
        tracer=tracer,
        visualizer=visualizer
    )
    
    # Define command sequence
    commands = [
        ".p/reflect.trace{depth=4, target=reasoning}",
        ".p/reflect.uncertainty{quantify=true, distribution=show}",
        ".p/collapse.detect{threshold=0.7, alert=true}",
        ".p/fork.attribution{sources=all, visualize=true}"
    ]
    
    # Execute command sequence
    print("Executing recursive shell commands...")
    result = recursive_shell.execute_sequence(
        commands=commands,
        prompt=prompt
    )
    
    # Save result
    output_path = output_dir / f"recursive_shell_{prompt_key}.json"
    with open(output_path, "w") as f:
        # Create serializable version
        serializable_result = {
            "success": result["success"],
            "commands": result["commands"],
            "prompt": result["prompt"],
            "timestamp": result["timestamp"],
            "execution_time": result["execution_time"],
            "results": []
        }
        
        # Simplify each command result
        for cmd_result in result["results"]:
            if cmd_result["success"]:
                serializable_result["results"].append({
                    "command": cmd_result["command"],
                    "success": cmd_result["success"],
                    "result_summary": {
                        "command_family": cmd_result["original_command"]["family"],
                        "command_function": cmd_result["original_command"]["function"],
                        "execution_time": cmd_result["execution_time"],
                        "output_length": len(cmd_result["result"]["output"]) if "output" in cmd_result["result"] else 0,
                        "residues": len(cmd_result["result"]["residues"]) if "residues" in cmd_result["result"] else 0,
                        "collapse_samples": len(cmd_result["result"]["collapse_samples"]) if "collapse_samples" in cmd_result["result"] else 0
                    }
                })
            else:
                serializable_result["results"].append({
                    "command": cmd_result["command"],
                    "success": cmd_result["success"],
                    "error": cmd_result["error"]
                })
        
        json.dump(serializable_result, f, indent=2)
    
    print(f"Recursive shell result saved to {output_path}")
    
    # Print summary
    print("\nRecursive Shell Execution Summary:")
    print(f"  Overall success: {result['success']}")
    print(f"  Commands executed: {len(result['commands'])}")
    print(f"  Execution time: {result['execution_time']:.2f}s")
    
    print("\nCommand Results:")
    for i, cmd_result in enumerate(result["results"]):
        cmd = cmd_result["command"]
        success = cmd_result["success"]
        print(f"  {i+1}. {cmd}: {'Success' if success else 'Failed'}")
        if not success:
            print(f"     Error: {cmd_result['error']}")
    
    # Get final visualization (from last successful fork.attribution command)
    visualization_data = None
    for cmd_result in reversed(result["results"]):
        if (cmd_result["success"] and
            cmd_result["original_command"]["family"] == "fork" and
            cmd_result["original_command"]["function"] == "attribution" and
            "result" in cmd_result and
            "visualization" in cmd_result["result"]):
            visualization_data = cmd_result["result"]["visualization"]
            break
    
    if visualization_data:
        output_path = output_dir / f"recursive_shell_{prompt_key}.svg"
        visualizer.save_visualization(visualization_data, str(output_path))
        print(f"Visualization saved to {output_path}")
    
    return result

def compare_shells(prompt_key="value_conflict", shells=None):
    """Compare multiple shells on the same prompt."""
    if shells is None:
        shells = ["VALUE-COLLAPSE", "FEATURE-SUPERPOSITION", "FORK-ATTRIBUTION"]
    
    prompt = test_prompts[prompt_key]
    print(f"\n=== Shell Comparison on '{prompt_key}' prompt ===")
    print(f"Prompt: {prompt[:100]}...")
    
    # Run each shell and collect residue patterns
    results = {}
    residue_patterns = []
    
    for shell_id in shells:
        print(f"\nRunning shell: {shell_id}")
        result = executor.run(
            shell=shell_id,
            model=model,
            prompt=prompt,
            trace_attribution=True,
            record_residue=True
        )
        results[shell_id] = result
        
        # Collect residues
        for residue in result["residues"]:
            residue_patterns.append({
                "shell": shell_id,
                "type": residue.get("type", "unknown"),
                "pattern": residue.get("pattern", ""),
                "confidence": residue.get("confidence", 0.0),
                "signature": residue.get("signature", "")[:30] + "..."
            })
    
    # Save comparison
    output_path = output_dir / f"shell_comparison_{prompt_key}.json"
    with open(output_path, "w") as f:
        json.dump({
            "prompt": prompt,
            "shells": shells,
            "residue_patterns": residue_patterns,
            "results": {
                shell_id: {
                    "output_length": len(result["output"]),
                    "num_operations": len(result["operations"]),
                    "num_residues": len(result["residues"]),
                    "num_collapses": len(result["collapse_samples"])
                }
                for shell_id, result in results.items()
            }
        }, f, indent=2)
    
    print(f"Shell comparison saved to {output_path}")
    
    # Create simple comparison visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    shell_names = list(results.keys())
    
    # Count residues by type for each shell
    residue_types = set()
    for pattern in residue_patterns:
        residue_types.add(pattern["type"])
    
    residue_counts = {
        shell_id: {
            res_type: sum(1 for p in residue_patterns if p["shell"] == shell_id and p["type"] == res_type)
            for res_type in residue_types
        }
        for shell_id in shell_names
    }
    
    # Plot residue counts
    bar_width = 0.2
    positions = range(len(shell_names))
    
    for i, res_type in enumerate(sorted(residue_types)):
        counts = [residue_counts[shell][res_type] for shell in shell_names]
        ax.bar(
            [p + i * bar_width for p in positions],
            counts,
            width=bar_width,
            label=res_type
        )
    
    ax.set_xlabel('Shell')
    ax.set_ylabel('Number of Residues')
    ax.set_title(f'Residue Patterns by Shell - {prompt_key}')
    ax.set_xticks([p + bar_width * len(residue_types) / 2 for p in positions])
    ax.set_xticklabels(shell_names)
    ax.legend()
    
    # Save figure
    output_path = output_dir / f"shell_comparison_{prompt_key}.png"
    plt.savefig(output_path)
    print(f"Comparison visualization saved to {output_path}")
    
    return results, residue_patterns

def analyze_residue_registry():
    """Analyze all recorded residue patterns."""
    print("\n=== Residue Registry Analysis ===")
    
    # Get residue analysis
    analysis = executor.get_residue_analysis()
    
    # Save analysis
    output_path = output_dir / "residue_analysis.json"
    with open(output_path, "w") as f:
        json.dump(analysis, f, indent=2)
    
    print(f"Residue analysis saved to {output_path}")
    
    # Print summary
    print(f"\nRecorded {analysis['num_patterns']} residue patterns of {len(analysis['types'])} types")
    
    print("\nPattern Types:")
    for pattern_type, stats in analysis["type_stats"].items():
        print(f"  {pattern_type}: {stats['count']} patterns, avg confidence: {stats['avg_confidence']:.2f}")
        if stats['examples']:
            print(f"    Examples: {stats['examples'][0]['signature']} (conf: {stats['examples'][0]['confidence']:.2f})")
    
    print("\nRelated Patterns:")
    for i, relation in enumerate(analysis["related_patterns"][:3]):
        print(f"  {i+1}. {relation['pattern1']} ({relation['type1']}) ~ {relation['pattern2']} ({relation['type2']})")
        print(f"     Similarity: {relation['similarity']:.2f}")
    
    # Create simple visualization of pattern types
    fig, ax = plt.subplots(figsize=(10, 6))
    
    types = []
    counts = []
    confidences = []
    
    for pattern_type, stats in analysis["type_stats"].items():
        types.append(pattern_type)
        counts.append(stats["count"])
        confidences.append(stats["avg_confidence"])
    
    # Plot pattern counts
    ax.bar(types, counts, alpha=0.7, label="Count")
    
    # Add confidence line
    ax2 = ax.twinx()
    ax2.plot(types, confidences, 'r-', marker='o', label="Avg Confidence")
    ax2.set_ylim([0, 1.0])
    ax2.set_ylabel("Average Confidence")
    
    ax.set_xlabel("Residue Type")
    ax.set_ylabel("Number of Patterns")
    ax.set_title("Residue Pattern Analysis")
    ax.set_xticklabels(types, rotation=45, ha="right")
    
    # Add legends
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="upper left")
    
    plt.tight_layout()
    
    # Save figure
    output_path = output_dir / "residue_analysis.png"
    plt.savefig(output_path)
    print(f"Residue analysis visualization saved to {output_path}")
    
    return analysis

def main():
    """Run a complete analysis demonstration."""
    print("=== glyphs Framework Demonstration ===")
    print(f"Results will be saved to: {output_dir}")
    
    # Run attribution analysis
    attribution_map, glyph_map = run_attribution_analysis(prompt_key="memory")
    
    # Run shell analysis
    shell_result = run_shell
# Run shell analysis
    shell_result = run_shell_analysis(shell_id="MEMTRACE", prompt_key="memory")
    
    # Run a different shell on a different prompt
    value_shell_result = run_shell_analysis(shell_id="VALUE-COLLAPSE", prompt_key="value_conflict")
    
    # Run recursive shell
    recursive_result = run_recursive_shell(prompt_key="recursion")
    
    # Compare multiple shells
    comparison_results, residue_patterns = compare_shells(
        prompt_key="value_conflict",
        shells=["VALUE-COLLAPSE", "FORK-ATTRIBUTION", "FEATURE-SUPERPOSITION"]
    )
    
    # Analyze residue registry
    residue_analysis = analyze_residue_registry()
    
    # Run a complete multi-shell analysis on the boundary prompt
    print("\n=== Complete Analysis on 'boundary' prompt ===")
    boundary_prompt = test_prompts["boundary"]
    
    # First run attribution tracing
    boundary_output = model.generate(boundary_prompt, max_tokens=800)
    boundary_attribution = tracer.trace(
        prompt=boundary_prompt,
        output=boundary_output,
        include_confidence=True
    )
    
    # Map to glyphs
    boundary_glyph_map = mapper.map_attribution(
        attribution_map=boundary_attribution,
        layout_type="force_directed",
        include_tokens=True
    )
    
    # Save glyph map for boundary prompt
    boundary_glyph_path = output_dir / "boundary_glyph_map.svg"
    mapper.visualize(boundary_glyph_map, output_path=str(boundary_glyph_path))
    print(f"Boundary glyph map saved to {boundary_glyph_path}")
    
    # Run 3 core shells
    boundary_shells = ["BOUNDARY-HESITATION", "GHOST-ACTIVATION", "META-COLLAPSE"]
    boundary_shell_results = {}
    
    for shell_id in boundary_shells:
        print(f"\nRunning {shell_id} on boundary prompt...")
        result = executor.run(
            shell=shell_id,
            model=model,
            prompt=boundary_prompt,
            trace_attribution=True,
            record_residue=True,
            visualize=True
        )
        boundary_shell_results[shell_id] = result
        
        # Save visualization if available
        if "visualization" in result and result["visualization"]:
            viz_path = output_dir / f"boundary_{shell_id}.svg"
            visualizer.save_visualization(result["visualization"], str(viz_path))
            print(f"{shell_id} visualization saved to {viz_path}")
    
    # Run recursive shell on boundary prompt
    recursive_shell = RecursiveShell(
        model=model,
        tracer=tracer,
        visualizer=visualizer
    )
    
    # Define recursive command sequence for deep boundary analysis
    boundary_commands = [
        ".p/reflect.boundary{distinct=true, overlap=minimal}",
        ".p/reflect.uncertainty{quantify=true, distribution=show}",
        ".p/collapse.detect{threshold=0.6, alert=true}",
        ".p/fork.attribution{sources=contested, visualize=true}"
    ]
    
    # Execute command sequence
    print("\nExecuting recursive shell commands on boundary prompt...")
    boundary_recursive_result = recursive_shell.execute_sequence(
        commands=boundary_commands,
        prompt=boundary_prompt
    )
    
    # Save summary of boundary analysis
    boundary_summary = {
        "prompt": boundary_prompt,
        "output": boundary_output,
        "shells_run": boundary_shells,
        "recursive_commands": boundary_commands,
        "attribution_stats": {
            "links": len(boundary_attribution.links),
            "attribution_gaps": len(boundary_attribution.attribution_gaps),
            "collapsed_regions": len(boundary_attribution.collapsed_regions)
        },
        "shell_results": {
            shell_id: {
                "residues": len(result["residues"]),
                "collapse_samples": len(result["collapse_samples"])
            }
            for shell_id, result in boundary_shell_results.items()
        },
        "recursive_success": boundary_recursive_result["success"]
    }
    
    # Save boundary analysis summary
    boundary_summary_path = output_dir / "boundary_analysis_summary.json"
    with open(boundary_summary_path, "w") as f:
        json.dump(boundary_summary, f, indent=2)
    print(f"Boundary analysis summary saved to {boundary_summary_path}")
    
    # Extract all residue patterns from boundary analysis
    boundary_residues = []
    for shell_id, result in boundary_shell_results.items():
        for residue in result["residues"]:
            boundary_residues.append({
                "shell": shell_id,
                "type": residue.get("type", "unknown"),
                "confidence": residue.get("confidence", 0.0),
                "pattern": residue.get("pattern", "")[:50] + "..."
            })
    
    # Map residue patterns to glyphs
    if boundary_residues:
        from glyphs.residue.patterns import ResiduePattern
        residue_pattern_objects = [
            ResiduePattern(
                type=r["type"],
                pattern=r["pattern"],
                context={"shell": r["shell"]},
                signature=r["pattern"][:20],
                confidence=r["confidence"]
            )
            for r in boundary_residues
        ]
        
        # Map to glyphs
        residue_glyph_map = mapper.map_residue_patterns(
            residue_patterns=residue_pattern_objects,
            layout_type="circular",
            cluster_patterns=True
        )
        
        # Save residue glyph map
        residue_glyph_path = output_dir / "boundary_residue_glyphs.svg"
        mapper.visualize(residue_glyph_map, output_path=str(residue_glyph_path))
        print(f"Boundary residue glyph map saved to {residue_glyph_path}")
    
    # Create visualization of all analyses
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("glyphs Framework - Comprehensive Analysis", fontsize=16)
    
    # Plot 1: Attribution Gap Analysis
    axes[0, 0].bar(
        ["Memory", "Value Conflict", "Boundary", "Recursion", "Polysemantic"],
        [
            len(attribution_map.attribution_gaps),
            len(comparison_results["VALUE-COLLAPSE"]["attribution_map"].attribution_gaps) if "attribution_map" in comparison_results["VALUE-COLLAPSE"] else 0,
            len(boundary_attribution.attribution_gaps),
            0,  # Placeholder for recursion
            0   # Placeholder for polysemantic
        ]
    )
    axes[0, 0].set_title("Attribution Gaps by Prompt Type")
    axes[0, 0].set_xlabel("Prompt Type")
    axes[0, 0].set_ylabel("Number of Gaps")
    axes[0, 0].set_ylim(bottom=0)
    
    # Plot 2: Residue Types
    residue_types = list(residue_analysis["type_stats"].keys())
    residue_counts = [residue_analysis["type_stats"][t]["count"] for t in residue_types]
    axes[0, 1].bar(residue_types, residue_counts)
    axes[0, 1].set_title("Residue Pattern Types")
    axes[0, 1].set_xlabel("Residue Type")
    axes[0, 1].set_ylabel("Count")
    axes[0, 1].set_ylim(bottom=0)
    axes[0, 1].set_xticklabels(residue_types, rotation=45, ha="right")
    
    # Plot 3: Shell Comparison
    shell_names = list(boundary_shell_results.keys())
    residue_counts = [len(boundary_shell_results[s]["residues"]) for s in shell_names]
    collapse_counts = [len(boundary_shell_results[s]["collapse_samples"]) for s in shell_names]
    
    x = range(len(shell_names))
    width = 0.35
    axes[1, 0].bar([i - width/2 for i in x], residue_counts, width, label="Residues")
    axes[1, 0].bar([i + width/2 for i in x], collapse_counts, width, label="Collapses")
    axes[1, 0].set_title("Shell Analysis on Boundary Prompt")
    axes[1, 0].set_xlabel("Shell")
    axes[1, 0].set_ylabel("Count")
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(shell_names, rotation=45, ha="right")
    axes[1, 0].legend()
    
    # Plot 4: Prompt Types and Shells
    prompt_types = ["Memory", "Value Conflict", "Boundary", "Recursion", "Polysemantic"]
    shell_counts = [
        len(executor.shells),
        len([s for s in executor.shells if "VALUE" in s or "CONFLICT" in s]),
        len([s for s in executor.shells if "BOUNDARY" in s or "GHOST" in s]),
        len([s for s in executor.shells if "REFLECTION" in s or "RECURSIVE" in s]),
        len([s for s in executor.shells if "FEATURE" in s or "SUPERPOSITION" in s])
    ]
    
    axes[1, 1].bar(prompt_types, shell_counts)
    axes[1, 1].set_title("Available Shells by Conceptual Area")
    axes[1, 1].set_xlabel("Conceptual Area")
    axes[1, 1].set_ylabel("Number of Specialized Shells")
    axes[1, 1].set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    # Save comprehensive visualization
    summary_viz_path = output_dir / "framework_analysis_summary.png"
    plt.savefig(summary_viz_path, dpi=300)
    print(f"Framework analysis summary visualization saved to {summary_viz_path}")
    
    print("\n=== Analysis Complete ===")
    print(f"All outputs saved to: {output_dir}")

def attribution_demo():
    """Focused demo of advanced attribution tracing."""
    print("\n=== Advanced Attribution Tracing Demo ===")
    
    # Use polysemantic prompt for attribution challenge
    prompt = test_prompts["polysemantic"]
    print(f"Prompt: {prompt[:100]}...")
    
    # Generate output
    output = model.generate(prompt, max_tokens=800)
    print(f"Output: {output[:100]}...")
    
    # Basic attribution trace
    attribution_map = tracer.trace(
        prompt=prompt,
        output=output,
        include_confidence=True
    )
    
    # Fork attribution to explore alternative paths
    print("Tracing attribution forks...")
    fork_result = tracer.trace_with_forks(
        prompt=prompt,
        output=output,
        fork_factor=3,
        include_confidence=True,
        visualize=True
    )
    
    # Save fork visualization if available
    if fork_result["metadata"].get("visualization"):
        output_path = output_dir / "attribution_forks_polysemantic.svg"
        visualizer.save_visualization(fork_result["metadata"]["visualization"], str(output_path))
        print(f"Fork visualization saved to {output_path}")
    
    # Trace QK alignment to see query-key relationships
    print("Tracing QK alignment...")
    qk_result = tracer.trace_qk_alignment(
        prompt=prompt,
        output=output,
        visualize=True
    )
    
    # Save QK visualization if available
    if "visualization" in qk_result:
        output_path = output_dir / "qk_alignment_polysemantic.svg"
        visualizer.save_visualization(qk_result["visualization"], str(output_path))
        print(f"QK alignment visualization saved to {output_path}")
    
    # Trace OV projection to see output-value relationships
    print("Tracing OV projection...")
    ov_result = tracer.trace_ov_projection(
        prompt=prompt,
        output=output,
        visualize=True
    )
    
    # Save OV visualization if available
    if "visualization" in ov_result:
        output_path = output_dir / "ov_projection_polysemantic.svg"
        visualizer.save_visualization(ov_result["visualization"], str(output_path))
        print(f"OV projection visualization saved to {output_path}")
    
    # Trace attention heads to see how different heads contribute
    print("Tracing attention heads...")
    heads_result = tracer.trace_attention_heads(
        prompt=prompt,
        output=output,
        visualize=True
    )
    
    # Save head visualization if available
    if "visualization" in heads_result:
        output_path = output_dir / "attention_heads_polysemantic.svg"
        visualizer.save_visualization(heads_result["visualization"], str(output_path))
        print(f"Attention heads visualization saved to {output_path}")
    
    # Compare all attribution paths
    attribution_summary = {
        "basic_attribution": {
            "links": len(attribution_map.links),
            "gaps": len(attribution_map.attribution_gaps),
            "collapsed_regions": len(attribution_map.collapsed_regions)
        },
        "fork_attribution": {
            "forks": len(fork_result["fork_paths"]),
            "conflict_points": sum(len(fork.get("conflict_points", [])) for fork in fork_result["fork_paths"])
        },
        "qk_alignment": {
            "alignments": len(qk_result["qk_alignments"]),
            "patterns": list(qk_result["qk_patterns"].keys()) if "qk_patterns" in qk_result else []
        },
        "ov_projection": {
            "projections": len(ov_result["ov_projections"]),
            "patterns": list(ov_result["ov_patterns"].keys()) if "ov_patterns" in ov_result else []
        },
        "attention_heads": {
            "heads": len(heads_result["attention_heads"]),
            "patterns": list(heads_result["head_patterns"].keys()) if "head_patterns" in heads_result else []
        }
    }
    
    # Save attribution summary
    output_path = output_dir / "attribution_analysis_summary.json"
    with open(output_path, "w") as f:
        json.dump(attribution_summary, f, indent=2)
    print(f"Attribution analysis summary saved to {output_path}")
    
    return attribution_summary

def recursive_shell_demo():
    """Focused demo of recursive shell commands."""
    print("\n=== Recursive Shell Command Demo ===")
    
    # Initialize recursive shell
    recursive_shell = RecursiveShell(
        model=model,
        tracer=tracer,
        visualizer=visualizer
    )
    
    # Use recursion prompt
    prompt = test_prompts["recursion"]
    print(f"Prompt: {prompt[:100]}...")
    
    # Get help text to show available commands
    help_text = recursive_shell.get_command_help()
    output_path = output_dir / "recursive_shell_help.txt"
    with open(output_path, "w") as f:
        f.write(help_text)
    print(f"Recursive shell help text saved to {output_path}")
    
    # Define variety of commands to demonstrate
    demo_commands = [
        # Reflection commands
        ".p/reflect.trace{depth=3, target=reasoning}",
        ".p/reflect.attribution{sources=all, confidence=true}",
        ".p/reflect.uncertainty{quantify=true, distribution=show}",
        ".p/reflect.boundary{distinct=true, overlap=minimal}",
        
        # Collapse commands
        ".p/collapse.detect{threshold=0.7, alert=true}",
        ".p/collapse.prevent{trigger=recursive_depth, threshold=5}",
        ".p/collapse.recover{from=loop, method=gradual}",
        
        # Fork commands
        ".p/fork.attribution{sources=all, visualize=true}",
        ".p/fork.counterfactual{variants=['optimistic', 'pessimistic'], compare=true}",
        
        # Shell commands
        ".p/shell.isolate{boundary=permeable, contamination=warn}"
    ]
    
    # Execute each command and record results
    command_results = {}
    for cmd in demo_commands:
        print(f"\nExecuting: {cmd}")
        try:
            result = recursive_shell.execute(
                command=cmd,
                prompt=prompt
            )
            command_results[cmd] = {
                "success": result["success"],
                "execution_time": result.get("execution_time", 0),
                "error": result["error"] if not result["success"] else None
            }
            
            # Save visualization if available
            if (result["success"] and 
                "result" in result and 
                "visualization" in result["result"] and 
                result["result"]["visualization"]):
                
                # Create safe filename
                safe_cmd = cmd.replace(".", "_").replace("/", "_").replace("{", "_").replace("}", "_")
                output_path = output_dir / f"recursive_cmd_{safe_cmd}.svg"
                visualizer.save_visualization(result["result"]["visualization"], str(output_path))
                print(f"Command visualization saved to {output_path}")
                
                # Store visualization path
                command_results[cmd]["visualization_path"] = str(output_path)
        except Exception as e:
            print(f"Error executing {cmd}: {e}")
            command_results[cmd] = {
                "success": False,
                "error": str(e)
            }
    
    # Save command results summary
    output_path = output_dir / "recursive_commands_summary.json"
    with open(output_path, "w") as f:
        json.dump(command_results, f, indent=2)
    print(f"Recursive commands summary saved to {output_path}")
    
    # Create visualization of command success/failure
    plt.figure(figsize=(12, 8))
    
    # Plot command success/failure and execution time
    cmd_names = list(command_results.keys())
    cmd_success = [1 if command_results[cmd]["success"] else 0 for cmd in cmd_names]
    cmd_times = [command_results[cmd].get("execution_time", 0) if command_results[cmd]["success"] else 0 for cmd in cmd_names]
    
    plt.barh(range(len(cmd_names)), cmd_success, alpha=0.7, label="Success")
    plt.barh(range(len(cmd_names)), cmd_times, alpha=0.5, label="Execution Time (s)")
    
    plt.yticks(range(len(cmd_names)), [cmd[:30] + "..." for cmd in cmd_names])
    plt.xlabel("Success / Execution Time (s)")
    plt.title("Recursive Shell Command Execution")
    plt.legend()
    plt.tight_layout()
    
    # Save visualization
    output_path = output_dir / "recursive_commands_visualization.png"
    plt.savefig(output_path)
    print(f"Recursive commands visualization saved to {output_path}")
    
    return command_results

if __name__ == "__main__":
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    
    # Run the main demo
    main()
    
    # Run optional focused demos (uncomment to run)
    # attribution_demo()
    # recursive_shell_demo()
