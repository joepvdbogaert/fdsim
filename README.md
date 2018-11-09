# fdsim
The fdsim package is a simulation engine for incidents and responses of fire departments in The Netherlands. It is being developed as part of a collaboration between Fire Department Amsterdam-Amstelland (FDAA) and Jheronimus Academy of Data Science (JADS). The overall objective of FDAA is to reduce their response times. This package facilitates in strategic, tactical, and operational decision making by providing ways to evaluate response times for different scenarios.

Some use cases of fdsim are to:
- evaluate candidate locations for a new fire station
- evaluate effects of closing a station
- evaluate alternative vehicle-to-station allocations
- evaluate effect of purchasing an extra vehicle (and decide its position)
- evaluate performance gain from dynamically relocating vehicles and compare different algorithms
- evaluate different dispatch rules
- find weak spots in current system

## Installation
To install the package, follow the following steps:
- Download or clone the repository
- Open a terminal and move to the root of the repository on your local machine: `cd path/to/fdsim`
- In the root folder of fdsim, install with pip: `pip install .`

The package should now be installed and can be imported everywhere just like any other Python package.

## Notes
fdsim is under development and currently specifically built to deal with the data that is available at FDAA. A generalization step might be made when this is considered useful by other fire departments.

