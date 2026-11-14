"""Ground stage: cropped scan -> above-ground points in the ground frame.

* ``methods/``    the pluggable ground-removal methods (``remove_ground``)
* ``ransac.py``   the RANSAC plane kernels the default method uses
* ``alignment.py``  rotation that puts the fitted plane at z = 0
* ``separation.py`` the stage entry point (``separate_ground``)
"""
