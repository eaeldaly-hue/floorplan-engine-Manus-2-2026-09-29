"""Unified architectural layer: typed architectural elements -> topology -> API records.

The recognition engines (legacy structure, Plan Model, Architectural Cleaner) each understand
part of the drawing. This package turns their typed elements into one representation and the
records the API reports, instead of re-detecting what an earlier stage already understood
(e.g. openings re-found from pixels after the cleaner typed and sealed them).
"""
