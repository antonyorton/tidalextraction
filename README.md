# Tidalextraction
Modules to deal with extraction of tidal signals from groundwater levels or to extract desired frequencies from a pure tide

## Purpose
To enable filtering of tidal frequencies from groundwater level data
To enable tidal prediction from previous tidal observations

# Included
BrisbaneBar2016.csv and BrisbaneBar2017.csv are sample tidal data from https://www.msq.qld.gov.au/Tides/Open-data
The first file can be used to extract freqiencies and the second to test prediction

tide_constituents.csv is a set of tidal constituent frequencies obtained from NOAA Anchorage station https://tidesandcurrents.noaa.gov/harcon.html?id=9455920
These are based on the moon and sun and so the same for every station. The things that differ are the amplitudes and phases of each frequency which is what the 
function extract_amplitudes_and_phase will pull out
