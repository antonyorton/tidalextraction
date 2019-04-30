## 1. Tidalextraction
Modules to deal with extraction of tidal signals from groundwater levels or to extract desired frequencies from a pure tide
  "Optimise_tides.py" is the most robust, using scipy.curve_fit. 
  "extract_tidal_constituents.py" shows the full least squares details although has issue of leaving a global phase shift which I have not    yet worked out how to fix

#### Purpose
To enable filtering of tidal frequencies from groundwater level data
To enable tidal prediction from previous tidal observations

#### Included sample datasets
BrisbaneBar2016.csv and BrisbaneBar2017.csv are sample tidal data from https://www.msq.qld.gov.au/Tides/Open-data
The first file can be used to extract freqiencies and the second to test prediction

tide_constituents.csv is a set of tidal constituent frequencies obtained from NOAA Anchorage station https://tidesandcurrents.noaa.gov/harcon.html?id=9455920
These are based on the moon and sun and so the same for every station. The things that differ are the amplitudes and phases of each frequency which is what the 
function extract_amplitudes_and_phase will pull out

## 2. Parameter estimation
A program for fitting Theis or Leaky Aquifer curves to pumping test groundwater responses and which has been tested on a large number of actual pumping tests.

The theory can be found in this excellent reference: https://www.hydrology.nl/images/docs/dutch/key/Kruseman_and_De_Ridder_2000.pdf
