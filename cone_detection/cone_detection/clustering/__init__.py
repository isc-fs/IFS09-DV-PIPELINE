"""Clustering stage: above-ground points -> DBSCAN labels.

* ``stage.py``       the stage entry point (``cluster_scan``): pre-filters, then DBSCAN
* ``prefilters.py``  what is dropped before DBSCAN (flat cells, tall columns, size guard)
* ``dbscan.py``      the DBSCAN backends (``cluster_points``)
"""
