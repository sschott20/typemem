"""Memory visualization tools."""
from typemem.viz.server import VizServer


def start_viz(manager, engine=None, injector=None, port=8811,
              open_browser=True, db_presets=None):
    """Start the visualization server.

    Args:
        manager: MemoryManager instance
        engine: Optional ConsolidationEngine (for consolidation event tracking)
        injector: Optional MemoryInjector (for injection event tracking)
        port: HTTP server port
        open_browser: Whether to open a browser tab
        db_presets: Optional dict of {name: path} for database presets

    Returns:
        VizServer instance (call .stop() when done).
    """
    server = VizServer(manager, engine=engine, injector=injector, port=port,
                       db_presets=db_presets)
    server.start()
    if open_browser:
        import webbrowser
        webbrowser.open(f"http://localhost:{server.port}")
    return server
