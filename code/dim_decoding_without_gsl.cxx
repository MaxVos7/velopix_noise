// ======================================
//
// Libraries
//
// ======================================
#include <iostream> // Needed to use std namespace + user input
#include <math.h>   // Needed to use pow and sqrt
#include <stdint.h> // Needed to use uint8_t
#include <stdio.h>  // Needed to use printf
#include <string>
#include <sstream>  // string splitting
#include <vector>   // string splitting

using namespace std;

// ======================================
//
// Global variables
//
// ======================================

int mean_trim0[256*256];

// ======================================
//
// Helper Functions
//
// ======================================

/* Goal: Generic function to split strings into substrings
 */
vector<string> split(const string& s, char delimiter)
{
   vector<string> tokens;
   string token;
   istringstream tokenStream(s);
   while (getline(tokenStream, token, delimiter))
   {
      tokens.push_back(token);
   }
   return tokens;
}

/* Goal: Look-up table to convert read LFSR to number of hits
 * Input: lfsr value
 */
uint8_t velopix_mapping_LFSR_2_hits(uint8_t lfsr)
{
  switch (lfsr){
      case 0x00: return 0;
      case 0x01: return 1;
      case 0x02: return 2;
      case 0x03: return 7;
      case 0x04: return 3;
      case 0x05: return 13;
      case 0x06: return 8;
      case 0x07: return 27;
      case 0x08: return 4;
      case 0x09: return 33;
      case 0x0a: return 14;
      case 0x0b: return 36;
      case 0x0c: return 9;
      case 0x0d: return 49;
      case 0x0e: return 28;
      case 0x0f: return 19;
      case 0x10: return 5;
      case 0x11: return 25;
      case 0x12: return 34;
      case 0x13: return 17;
      case 0x14: return 15;
      case 0x15: return 53;
      case 0x16: return 37;
      case 0x17: return 55;
      case 0x18: return 10;
      case 0x19: return 46;
      case 0x1a: return 50;
      case 0x1b: return 39;
      case 0x1c: return 29;
      case 0x1d: return 42;
      case 0x1e: return 20;
      case 0x1f: return 57;
      case 0x20: return 63;
      case 0x21: return 6;
      case 0x22: return 12;
      case 0x23: return 26;
      case 0x24: return 32;
      case 0x25: return 35;
      case 0x26: return 48;
      case 0x27: return 18;
      case 0x28: return 24;
      case 0x29: return 16;
      case 0x2a: return 52;
      case 0x2b: return 54;
      case 0x2c: return 45;
      case 0x2d: return 38;
      case 0x2e: return 41;
      case 0x2f: return 56;
      case 0x30: return 62;
      case 0x31: return 11;
      case 0x32: return 31;
      case 0x33: return 47;
      case 0x34: return 23;
      case 0x35: return 51;
      case 0x36: return 44;
      case 0x37: return 40;
      case 0x38: return 61;
      case 0x39: return 30;
      case 0x3a: return 22;
      case 0x3b: return 43;
      case 0x3c: return 60;
      case 0x3d: return 21;
      case 0x3e: return 59;
      case 0x3f: return 58;
  }
  // Should not arrive here
  return 127;
}

  
// ======================================
//
// Primary Functions
//
// ======================================
/* Goal: Decode the raw lfsr data to hits and calculate the mean and width of the noise per pixel
 * Input: (1) raw data file
 */
int process_scan(string filename)
{
  // === Input ===
  // Parse filename for settings
  vector<string> tokens = split(filename, '_');
  int ntok = tokens.size();
  int dac_max = stoi(tokens[ntok-4]);
  int dac_step = stoi(tokens[ntok-3]);
  int npoints = stoi(tokens[ntok-2]);
  // Reconstruct filepath (i.e. remove .dat)
  string prefix = split(filename, '.')[0];

  // === Initialise ===
  // Store the scan data per pixel coordinate, going row per row
  // (0,0): 0 -> npoints
  // (0,1): npoints -> 2*npoints
  // (r,c): (256*r + c)*npoints ->
  uint8_t matrix[256*256*npoints];

  // === Load Data ===
  FILE *input_file = fopen(filename.c_str(), "r");
  char regVal[192*256];
  uint8_t lfsr = 0;
  int id = 0;

  // Loop over file
  for (int k=0; k<npoints; ++k) {
    // read complete matrix as that is also how it's written to file
    fread(regVal, sizeof(char), sizeof(regVal), input_file);

    // In C, bit-operations need to be done byte by byte
    // So handle 3 bytes = 4 pixels at a time
    for (int i=0; i<256; ++i) {
      for (int j=0; j<64; ++j) {
        id = (256*(4*j + 0) + i)*npoints + k;
        lfsr = (regVal[(3*j + 0) + 192*i] & 0xFC) >> 2;
        matrix[id] = velopix_mapping_LFSR_2_hits(lfsr);

        id = (256*(4*j + 1) + i)*npoints + k;
        lfsr = ((regVal[(3*j + 1) + 192*i] & 0xF0) >> 4) | ((regVal[(3*j + 0) + 192*i] & 0x03) << 4);
        matrix[id] = velopix_mapping_LFSR_2_hits(lfsr);

        id = (256*(4*j + 2) + i)*npoints + k;
        lfsr = ((regVal[(3*j + 2) + 192*i] & 0xC0) >> 6) | ((regVal[(3*j + 1) + 192*i] & 0x0F) << 2);
        matrix[id] = velopix_mapping_LFSR_2_hits(lfsr);

        id = (256*(4*j + 3) + i)*npoints + k;
        lfsr =  regVal[(3*j + 2) + 192*i] & 0x3F;
        matrix[id] = velopix_mapping_LFSR_2_hits(lfsr);
      }
    }
  }
  fclose(input_file);


  // === Process the Data ===
  
  // (0) study individual pixels
  string name_pix = prefix + "_Pixel_1_1.csv";
  FILE *file_pix = fopen(name_pix.c_str(), "w");
  int cdac;
  int i = 257; // pixel (1,1)
  for (int dac=0; dac<npoints; ++dac) {
    cdac = dac_max - dac*dac_step;
    fprintf(file_pix, "%04d, %02d\n", cdac, matrix[i*npoints + dac]);
  }
  fclose(file_pix);
  
  // (1) calculate mean of noise
  uint16_t arg_mean[256*256];
  string name_mean = prefix + "_Noise_Mean.csv";
  FILE *file_mean = fopen(name_mean.c_str(), "w");
  int nhits = 0;
  int imean = 0;

  for (int i=0; i<256*256; ++i) {
    imean = 0;
    nhits = 0;
    for (int dac=0; dac<npoints; ++dac) {
      imean += matrix[i*npoints + dac] * (dac_max - dac*dac_step);
      nhits += matrix[i*npoints + dac];
    }
    if (nhits>0) arg_mean[i] = 1.*imean/nhits;
    else arg_mean[i] = 0;
    
    if (i%256==255) fprintf(file_mean, "%04d\n", arg_mean[i]);
    else fprintf(file_mean, "%04d, ", arg_mean[i]);
  }
  fclose(file_mean);
  
  
  // (2) calculate width of noise
  string name_width = prefix + "_Noise_Width.csv";
  FILE *file_width = fopen(name_width.c_str(), "w");
  float iwidth = 0;
  float width = 0;
  
  for (int i=0; i<256*256; ++i) {
    iwidth = 0;
    nhits = 0;
    for (int dac=0; dac<npoints; ++dac) {
      iwidth += matrix[i*npoints + dac] * pow(dac_max - dac*dac_step - arg_mean[i],2);
      nhits += matrix[i*npoints + dac];
    }
    if (nhits>0) width = sqrt(1.*iwidth/(nhits-1));
    else width = 0;
      
    if (i%256==255) fprintf(file_width, "%05.2f\n", width);
    else fprintf(file_width, "%05.2f, ", width);
  }
  fclose(file_width);
  
  return 0;
}

/* Goal: Merge the mean and width output files
 * Input: (1) raw data file, (2) the number of files to merge
 */
int merge_scans(string prefix, int nmasks)
{
  // Initialise
  vector<string> tokens = split(prefix, '_');
  int ntok = tokens.size();
  string short_prefix = "";
  for (int i=0; i<ntok-6; ++i) short_prefix += tokens[i] + "_";
  short_prefix += tokens[ntok-4]; // Add which Trim

  char buffer[500];
  FILE *next_file;
  char next_line[2000];
  vector<string> next_vals;
  int id;

  FILE *file_merge;
  float matrix_merge[256*256];
  const int nMerge = 2;
  string who[nMerge] = {"Noise_Mean", "Noise_Width"};
  string formA[nMerge] = {"%04.0f\n", "%05.2f\n"};
  string formB[nMerge] = {"%04.0f, ", "%05.2f, "};

  for (int nt=0; nt<nMerge; ++nt) {
    // Clear
    for (int i=0; i<256*256; ++i) matrix_merge[i] = 0;

    // Loop overfiles
    cout << "[dim_equalisation] Merging " << who[nt] << " Files" << endl;
    for (int f=1; f<=nmasks; ++f) {
      sprintf(buffer, "%s%dof%d_%s.csv", prefix.c_str(), f, nmasks, who[nt].c_str());
      next_file = fopen(buffer, "r");
      id = 0;
    
      for (int l=0; l<256; ++l) {
        fgets(next_line, 2000, next_file);
        next_vals = split(next_line, ',');
        for (int val=0; val<256; ++val) {
          matrix_merge[id] += stof(next_vals[val]);
          id++;
        }
      }
      fclose(next_file);
    }

    // Save Noise Mean
    sprintf(buffer, "%s_%s.csv", short_prefix.c_str(), who[nt].c_str());
    file_merge = fopen(buffer, "w");
    for (int i=0; i<256*256; ++i) {
      if (i%256==255) fprintf(file_merge, formA[nt].c_str(), matrix_merge[i]);
      else fprintf(file_merge, formB[nt].c_str(), matrix_merge[i]);
    }
    fclose(file_merge);
  }

  return 0;
}

// ======================================
//
// Main
//
// ======================================
int main(int argc, char* argv[])
{
  // === Input ===
  if (argc==1) {
    cout << "[dim_equalisation] FAILED: please specify input file";
    return 0;
  }
  string filename = argv[1];

  // Parse filename for settings
  vector<string> tokens = split(filename, '_');
  int ntok = tokens.size();
  // Reconstruct filepath and velopix ID
  string prefix = "";
  for (int i=0; i<ntok-1; ++i) prefix += tokens[i] + "_";
  int nmasks = stoi(split(split(tokens[ntok-1], '.')[0], 'f')[1]);


  // === Calculate Mean and Width ===
  char buffer[500];
  for (int i=1; i<=nmasks; ++i) {
    sprintf(buffer, "%s%dof%d.dat", prefix.c_str(), i, nmasks);
    filename = buffer;
    cout << "[dim_equalisation] Analysing " << filename << endl;
    process_scan(filename);
  }


  // === Merge Files ===
  merge_scans(prefix, nmasks);
  
  return 0;
}
