"""Memory visualization tools."""
from typemem.viz.tracing import TracingStore, TracingSystem
from typemem.viz.server import VizServer


def start_viz(
    tracing_store: TracingStore,
    tracing_system: TracingSystem,
    port: int = 8811,
    open_browser: bool = True,
) -> VizServer:
    """Start the visualization server.

    Returns the VizServer instance (call .stop() when done).
    """
    server = VizServer(tracing_store, tracing_system, port=port)
    server.start()
    if open_browser:
        import webbrowser
        webbrowser.open(f"http://localhost:{server.port}")
    return server
