"""Fitting stage: labelled above-ground points -> accepted cones.

* ``extraction.py``  the stage entry point (``extract_cones``): gates, fit, books
* ``clusters.py``    labels -> per-cluster points and statistics
* ``gates.py``       which clusters are accepted, and why not
* ``fitters.py``     one cluster fitter per ``fit_backend``
* ``uncertainty.py`` the sigma_xy handed to SLAM
* cone models: ``cone_classes.py`` (the two FSAE cone shapes), ``template_fit.py``
  (production), ``closed_form_fits.py`` (warm start, collinear clusters),
  ``alternative_fits.py`` (two_param backend, VARPRO)
"""
