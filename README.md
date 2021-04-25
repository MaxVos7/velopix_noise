# Velopix study
## Initialisation
### Creating executables
Go to the ./code directory and run the following command
````commandline
make
````
Make sure you run this on a linux (or linux based) system.

This will create executables from the cxx files.
### Decoding data
To decode the .dat files located in the data directory run 
the following command in the ./code directory
````commandline
./dim_decoding full_path/data/Module1_VP0-0_Scan_Trim*_****_*_**_1of1.dat
````
This created several csv files containing the decoded data.
### Running equalisation process
In order to run the equalisation process make sure you have the decoded csv
files for the trim 0 and trim F. These will be used in the equalisation process.

To run the equalisation process go to the ./code directory and run the 
following command
````commandline
./dim_equalisation full_path/data/Module1_VP0-0 
````
This will created 6 csv files, containing all information about 
which pixels takes what trim level and which pixel is masked.