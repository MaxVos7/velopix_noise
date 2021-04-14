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



// ======================================
//
// Primary Functions
//
// ======================================
/* Goal: Load the pixel noise mean from file
 * Input: (1) data file, (2) matrix for means
 */
int load_mean(string filename, uint16_t* matrix)
{
  // Initialise
  FILE *next_file = fopen(filename.c_str(), "r");
  char next_line[2500];
  vector<string> next_vals;
  int id = 0;

  // Clear
  for (int i=0; i<256*256; ++i) matrix[i] = 0;

  for (int l=0; l<256; ++l) {
    fgets(next_line, 2500, next_file);
    next_vals = split(next_line, ',');
    for (int val=0; val<256; ++val) {
      matrix[id] += stoi(next_vals[val]);
      id++;
    }
  }
  fclose(next_file);

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
    cout << "[dim_equalisation] FAILED: please specify input file (prefix)";
    return 0;
  }
  string prefix = argv[1];
  int dacRange = 25; // Tuneable parameter

  // === Load Trim 0 and Trim F Means ===
  uint16_t mean_trim0[256*256];
  load_mean(prefix+"_Trim0_Noise_Mean.csv", mean_trim0);
  uint16_t mean_trimF[256*256];
  load_mean(prefix+"_TrimF_Noise_Mean.csv", mean_trimF);
  

  // === Calculate Target ===
  cout << "[dim_equalisation] Equalising" << endl;
  int glob_mean_trim0 = 0;
  int glob_mean_trimF = 0;
  int nhits = 0;
  for (int i=0; i<256*256; ++i) {
    if (mean_trim0[i]>0 && mean_trimF[i]>0) {
      glob_mean_trim0 += mean_trim0[i];
      glob_mean_trimF += mean_trimF[i];
      nhits++;
    }
  }
  if (nhits==0) {
    cout << "[dim_equalisation] FAILED: Threshold scan has empty output file" << endl;
    return 0;
  }
  glob_mean_trim0 /= nhits;
  glob_mean_trimF /= nhits;
  int target = (glob_mean_trim0 + glob_mean_trimF)/2;
  
  float glob_width_trim0 = 0;
  float glob_width_trimF = 0;
  for (int i=0; i<256*256; ++i) {
    if (mean_trim0[i]>0 && mean_trimF[i]>0) {
      glob_width_trim0 += pow(glob_mean_trim0 - mean_trim0[i], 2);
      glob_width_trimF += pow(glob_mean_trimF - mean_trimF[i], 2);
    }
  }
  glob_width_trim0 = sqrt(glob_width_trim0/(nhits-1));
  glob_width_trimF = sqrt(glob_width_trimF/(nhits-1));
  
  // === Calculate optimal trim ===
  string name_mask = prefix + "_Matrix_Mask.csv";
  FILE *file_mask = fopen(name_mask.c_str(), "w");
  string name_trim = prefix + "_Matrix_Trim.csv";
  FILE *file_trim = fopen(name_trim.c_str(), "w");
  string name_pred = prefix + "_TrimBest_Noise_Predict.csv";
  FILE *file_pred = fopen(name_pred.c_str(), "w");
  
  float trim_scale;
  int trim;
  int mask;
  int predict[256*256];
  int diff;
  long nmasked = 0;
  int achieved_mean = 0;
  float achieved_width = 0;
  
  for (int i=0; i<256*256; ++i) {
    trim_scale = 1.*(mean_trimF[i] - mean_trim0[i])/16;
    trim = round((target - mean_trim0[i])/trim_scale);
    mask = 0;
    predict[i] = mean_trim0[i] + round(trim*trim_scale);
    diff = fabs(predict[i] - target);
    if (mean_trim0[i]==0 || mean_trimF[i]==0 || trim>15 || trim<0 || diff>dacRange) {
      if (mean_trim0[i]==0 && mean_trimF[i]==0) mask = 1;
      else if (mean_trim0[i]==0) mask = 2;
      else if (mean_trimF[i]==0) mask = 3;
      else if (trim>15 || trim<0) mask = 4;
      else if (diff>dacRange) mask = 5;
      else mask = 6; // Should not happen

      trim = 0;
      predict[i] = 0;
      nmasked++;
    } else {
      achieved_mean += predict[i];
    }
    // Save results
    if (i%256==255) {
      fprintf(file_mask, "%d\n", mask);
      fprintf(file_trim, "%d\n", trim);
      fprintf(file_pred, "%04d\n", predict[i]);
    } else {
      fprintf(file_mask, "%d,", mask);
      fprintf(file_trim, "%d,", trim);
      fprintf(file_pred, "%04d, ", predict[i]);
    }
  }
  fclose(file_mask);
  fclose(file_trim);
  achieved_mean /= (256*256-nmasked);
  
  for (int i=0; i<256*256; ++i) {
    if (predict[i]>0) achieved_width += pow(predict[i] - achieved_mean, 2);
  }
  achieved_width = sqrt(achieved_width/(256*256-nmasked-1));
  
  cout << "[dim_equalisation] Summary" << endl;
  cout << "  Trim 0 distribution: " << glob_mean_trim0 << " +/- " << round(glob_width_trim0) << endl;
  cout << "  Trim F distribution: " << glob_mean_trimF << " +/- " << round(glob_width_trimF) << endl;
  cout << "  Equalisation Target: " << target << endl;
  char buffer[25];
  sprintf(buffer, "  Achieved: %d +/- %.1f", achieved_mean, achieved_width);
  cout << buffer << endl;
  cout << "  Masked Pixels: " << nmasked << endl;
  
  // === Test Pulse Pattern ===
  // for completeness
  string name_tp = prefix + "_Matrix_TP.csv";
  FILE *file_tp = fopen(name_tp.c_str(), "w");
  for (int i=0; i<256*256; ++i) {
    if (i==256*255)fprintf(file_tp, "1,");
    else if (i%256==255) fprintf(file_tp, "0\n");
    else fprintf(file_tp, "0,");
  }
  //for (int i=0; i<255; ++i) fprintf(file_tp, "1,");
  //fprintf(file_tp, "1\n");
  fclose(file_tp);
  
  return 0;
}
