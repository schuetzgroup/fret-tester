The fret-tester package
=======================

This is a software implementation of the smFRET kinetics analysis method
described in the Journal of Chemical Physics article “Kinetic analysis of
single molecule FRET transitions without trajectories” [(1)](#Schrangl2018),
which was used to perform the data anlysis for the article.

The documentation can be found at https://schuetzgroup.github.io/fret-tester/.
There is also a
[Jupyter notebook containing a tutorial](https://github.com/schuetzgroup/fret-tester/blob/master/Tutorial.ipynb).


Requirements
------------
- Python 3 (tested on version 3.6)
- numpy (tested on version 1.13)
- scipy (tested on version 1.0)
- matplotlib


Installation
------------
- Download source (either use `git clone` or download a snapshot zip and unpack
  it somewhere).
- Go to the topmost folder of the source and run `python setup.py install`
  or add the topmost folder to the Python path.


<a name="Schrangl2018"></a>(1) Schrangl, Lukas; Göhring, Janett; Schütz,
  Gerhard J. (2018):
  “Kinetic analysis of single molecule FRET transitions without trajectories.”
  In: The Journal of Chemical Physics, 148 (2018), H. 12, p. 123328.
  Available at: [DOI: 10.1063/1.5006038](https://doi.org/10.1063/1.5006038)
