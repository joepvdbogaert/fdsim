# What is FDSIM?

The `fdsim` package is a simulation engine for incidents and responses of fire departments in The Netherlands. It is being developed as part of a collaboration between Fire Department Amsterdam-Amstelland (FDAA) and Jheronimus Academy of Data Science (JADS). The overall objective of FDAA is to reduce their response times. This package facilitates in strategic, tactical, and operational decision making by providing ways to evaluate response times for different scenarios.

Some use cases of fdsim are to:
- evaluate candidate locations for a new fire station
- evaluate alternative vehicle-to-station allocations
- evaluate changes in crew availability (e.g., full time or part time crews)
- evaluate effects of closing a station
- evaluate effects of closing a station part of the day or week
- evaluate effects of having a station operated by part time crew part of the day or week
- evaluate effect of purchasing an extra vehicle (and decide its position)
- evaluate performance gain from dynamically relocating vehicles and compare different algorithms
- evaluate different dispatch rules
- find weak spots in current system

## For who is it?
`fdsim` was developed for the fire department Amsterdam-Amstelland, but uses data that is largely available for all fire departments in The Netherlands. If you work for another fire brigade, in or outside The Netherlands, and would like to make use of the package, please contact us to see if this is feasible.

## Installation
You can install `fdsim` right from the github page. Simply open a terminal and run:

```
pip install git+https://github.com/joepvdbogaert/fdsim.git
```

The package should now be installed and can be imported everywhere just like any other Python package.

## Dependencies
The minimal dependencies to run simulations are:
- scipy
- statsmodels
- numpy
- pandas
- scikit-learn

But this assumes you can provide some data, such as forecasts and routing information, by yourself. In order to make full use of the simulation engine, we recommend installing the following extras:
- osrm
- fbprophet

## Notes
fdsim is under development and currently specifically built to deal with the data that is available at FDAA. A generalization step might be made when this is considered useful by other fire departments.
