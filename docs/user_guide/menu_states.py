"""Module states whose menus hold commands the main window does not show when it opens.

Each function receives the module's main window and returns the window whose menu bar is then
read (it may be the same window in another mode). Their extra commands get their own sections
in the feature reference, explained in ``features/<module>.tsv`` like the others.
"""
from guide.features import menu_state


@menu_state("stats", "Output Viewer")
def stats_output_viewer(window):
    viewer = window.ctx.output()
    viewer.show()
    return viewer


@menu_state("graph", "Graph window active")
def graph_window(window):
    window.new_graph()
    return window


@menu_state("graph", "Matrix window active")
def matrix_window(window):
    window.new_matrix()
    return window
