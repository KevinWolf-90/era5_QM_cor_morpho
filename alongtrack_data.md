# Extraction of temperature, relative humidity,  wind components, and fraction of cloud cover from ERA5 #
 
## General ##
ERA5 data is extracted along IAGOS flight trajectories. Flights from the years 2015 to 2021 are used. Extraction is performed with the nearest neighbor method by selecting the temporally and spatially closest ERA5 grid point.
The code that was used to extract data from ERA 5 with 1 hour resolution:
[A01_extract__along_flightpath_1h.py](A01_extract__along_flightpath_1h.py)

## Filtering ##
After the extraction the data is filtered. Only data is kept that fulfills the following criteria:

<ul>
  <li> IAGOS temperature value available  </li>
  <li> IAGOS temperature flagged for good or limited  </li>
  <li> Absolute difference between temperature from aircraft avionic system and IAGOS temperature is smaller than 5 K  </li>

  <li> IAGOS relative humidity (w.r.t. liquid water and ice) value available  </li>
  <li> Relative humidity w.r.t. ice flagged for good or limited  </li>
  <li> Relative humidity w.r.t. liquid water is between 0 and 100 %  </li>
  <li> Relative humidity w.r.t. ice is between 0 and 170 %  </li>
</ul>




## Smoothing ##
Two data sets are stored: 1) The original data that represents the native resolution of IAGOS with a measurement every 4 seconds. 2) A smoothed data set to better match the spatial resolution of IAGOS with the spatial resolution of ERA5; with however still a value every 4 seconds. The smoothing is realized by applying a Gaussian filter. The characteristics of the Gaussian filter are specified in Wolf et al 2023. The filter is not run over the data flags and only over the following variables:

<ul>
  <li> Temperature from IAGOS  </li>
  <li> Termpature from the aircraft avionic system  </li>
  <li> Particle number concentration  </li>
  <li> Relative humidity w.r.t. liquid water  </li>
  <li> Relative humidity w.r.t. ice  </li>
</ul>


## Storage ##

Yearly files of IAGOS and ERA5 along-track data is stored in a *.npz file that can be pickled.


