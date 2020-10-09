Dendro-somatic coupling in L5 ACC pyramidal neurons
===================================================

This code allows reproduction of simulation results from [Mengual2020]_.


Running the code
----------------

To run, make sure all requirements are installed on your machine, clone this
repository and from its source directory run:

   .. code-block:: shell

      compilechannels channels/


Usage
-----

The optimize the parameters of the ion channels, run the `optimizer.optimizeModel()`
function from a python console.

Optimization results can be plotted with functions in `plot_optimization_results.py`.
The h-current analysis can be ran and plotted with `plot_h_analysis.py`.


Requirements
------------
python>=3.7.6
neat>=0.9.0
neuron>=7.7.2
bluepyopt>=1.9.3
numpy>=1.19.2
matplotlib>=3.3.2


References
----------
.. [Mengual2020] Ulisses Marti Mengual, Willem Wybo, Lotte Spierenburg, Mirko Santello, Walter Senn, and Thomas Nevian (2020) *Efficient low-pass dendro-somatic coupling in the apical dendrite of layer 5 pyramidal neurons in the anterior cingulate cortex*, In Press.