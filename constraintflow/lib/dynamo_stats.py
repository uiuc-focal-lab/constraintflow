"""Dynamo compile activity per run, from torch._dynamo.utils.counters (no logging, so nothing
is formatted inside the timer).

    frames_compiled       frames Dynamo compiled: first compiles and recompiles
    graphs_compiled       FX graphs handed to Inductor
    graph_breaks          graph breaks that split a frame (each creates a resume function)
    recompile_limit_hits  frames that hit config.recompile_limit and run eager from then on
"""

from torch._dynamo.utils import compilation_time_metrics, counters


def snapshot():
    #FIX FOR: counters["frames"] skips fullgraph=True compiles (the jit build); every compile times _compile.compile_inner.
    return {"frames_compiled": len(compilation_time_metrics.get("_compile.compile_inner", ())),
            "graphs_compiled": counters["stats"]["unique_graphs"],
            "graph_breaks": sum(counters["resumes"].values()),
            "recompile_limit_hits": sum(v for k, v in counters["unimplemented"].items()
                                        if k.endswith("_limit reached"))}


def since(before):
    """Counts since the snapshot `before`."""
    now = snapshot()
    return {k: now[k] - before[k] for k in now}
