"""
show_results.py
===============
Prints a table comparing every model you have run so far, and checks whether
adding context made a real difference.

Run this any time after you have trained some models.

    python show_results.py
"""

from engine.scoring import show_table

show_table()
