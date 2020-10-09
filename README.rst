Dendro-somatic coupling in L5 ACC pyramidal neurons
===================================================

This code allows reproduction of simulation results from [Mengual2020]_.


Running the code
----------------

To run, make sure all requirements are installed on your machine. Notably, this
code uses [BluePyOpt](https://bluepyopt.readthedocs.io/en/latest/) [VanGeit2016]_
and [NEAT](https://neatdend.readthedocs.io/en/latest/) [Wybo2020]_.

Clone this repository and compile the ion channels by running from its source directory:

   .. code-block:: shell

      compilechannels channels/


Usage
-----

The optimize the parameters of the ion channels, run the `optimizer.optimizeModel()`
function from a python console.

Optimization results can be plotted with functions in `plot_optimization_results.py`.
The h-current analysis can be ran and plotted with `plot_h_analysis.py`.

Check out `examples.py` for specific usage examples of the various optimization
or plot functions.


Requirements
------------
- python     >= 3.7.6
- neat       >= 0.9.0
- neuron     >= 7.7.2
- bluepyopt  >= 1.9.3
- numpy      >= 1.19.2
- matplotlib >= 3.3.2


References
----------
.. [Mengual2020] Ulisses Mengual, Willem Wybo, Lotte Spierenburg, Mirko Santello, Walter Senn, and Thomas Nevian (2020) *Efficient low-pass dendro-somatic coupling in the apical dendrite of layer 5 pyramidal neurons in the anterior cingulate cortex*, In Press.
.. [VanGeit2016] Werner Van Geit, Michael Gevaert, Giuseppe Chindemi, Christian Rössert, Jean-Denis Courcol, Eilif Muller, Felix Schürmann, Idan Segev, and Henry Markram (2016) *BluePyOpt: Leveraging Open Source Software and Cloud Infrastructure to Optimise Model Parameters in Neuroscience*, Front. Neuroinf.
.. [Wybo2020] Willem Wybo, Jakob Jordan, Benjamin Ellenberger, Ulisses Mengual, Thomas Nevian, and Walter Senn (2020) *Data-driven reduction of dendritic morphologies with preserved dendro-somatic responses*, bioRxiv preprint
