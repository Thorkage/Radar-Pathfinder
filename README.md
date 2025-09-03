# Radar-Pathfinder
[![Python Version](https://img.shields.io/badge/python-3.13-blue)](https://www.python.org/downloads/)

*Pathfinder* is a set of algorithms to automatically detect interfaces in surface-penetrating radar data (for snowy targets). It was developed for the UAV-mounted UWiBaSS system from NORCE Research AS in Tromsø ([Jenssen and Jacobsen (2020)](https://doi.org/10.1080/09205071.2020.1799871), [Jenssen and Jacobsen (2021)](https://www.mdpi.com/2072-4292/13/13/2610), [Jenssen et al (2024)](https://arc.lib.montana.edu/snow-science/objects/ISSW2024_P8.2.pdf)) for the purpose of automatically deriving snow depths (+ associated uncertainties) and stratigraphy. 

![pathfinder_demo](./pathfinder_demo_plot.png)

*Pathfinder* is presented in detail in the *Pathfinder_paper.pdf*-file (which is an internship report for my university) -- a more sophisticated publication is aimed for towards the end of 2025. To get started with using Pathfinder, have a look at the *running_Pathfinder.ipynb* notebook. All relevant information should be either there or in the pdf. If any questions arise: Let me know.

### Running it on your own GPR or radar data 
*Pathfinder* can be run on any 2D array and has been tested on various snow-measuring systems (also non-UAV-mounted)! Have a look at the *make_RADAR.ipynb* notebook to learn how to convert your data into something that *Pathfinder* can read.

### Requirements

*Pathfinder* is built with Python v3.13. Additional packages are listed in requirements.yml and can be installed using e.g. conda with:

```
(base) git clone https://github.com/Thorkage/Radar-Pathfinder
(base) cd ./Radar-Pathfinder
(base) conda env create -f requirements.yml
(base) conda activate pathfinder
```

