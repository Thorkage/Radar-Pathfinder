# Radar-Pathfinder
[![Python Version](https://img.shields.io/badge/python-3.13-blue)](https://www.python.org/downloads/)

*Pathfinder* is a set of algorithms to automatically detect interfaces in surface-penetrating radar data from snowy targets. It was developed for the UAV-mounted UWiBaSS system from NORCE (Norwegian Research Center in Tromsø) ([Jenssen and Jacobsen (2020)](https://doi.org/10.1080/09205071.2020.1799871), [Jenssen and Jacobsen (2021)](https://www.mdpi.com/2072-4292/13/13/2610), [Jenssen et al (2024)](https://arc.lib.montana.edu/snow-science/objects/ISSW2024_P8.2.pdf)) for the purpose of deriving snow depths and stratigraphy. 

*Pathfinder* is presented in detail in [ref]. Since the UWiBaSS system uses pickled dictionaries as the main data structure, *Pathfinder* is built around this exact object design. The object should contatin at minimum the (n,m) radar echogram, a fasttime axis of length n and a slowtime axis of length m. 

### Requirements

*Pathfinder* is built with python v3.13. Additional packages are listed in requirements.yml and can be installed using e.g. conda with:

```
conda env create -f environment.yml
```





