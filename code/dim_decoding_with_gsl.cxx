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

// Non-Linear Least Squares Fitting
#include <cmath>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_multifit_nlinear.h>

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
// GSL Fitting Functions
//
// ======================================

struct data {
  int n;
  double *dac;
  double *hits;
};

int func (const gsl_vector * x, void *data, gsl_vector * f)
{
  int n = ((struct data *)data)->n;
  double *dac = ((struct data *)data)->dac;
  double *hits = ((struct data *)data)->hits;

  // Positive definite parameters
  double norm = fabs( gsl_vector_get (x, 0) );
  double mu = fabs( gsl_vector_get (x, 1) );
  double sigma = fabs( gsl_vector_get (x, 2) );
  int i;

  for (i = 0; i < n; i++)
    {
      /* Model g = norm/(sigma * sqrt(2 * pi)) * exp( - (x - mu)**2 / (2 * sigma**2) ) */
      double gauss = norm/(sigma * sqrt(4*asin(1))) * exp( -0.5 * pow((dac[i] - mu)/sigma, 2) );
      double Theo = fmin(63, gauss);
      if (Theo>0) gsl_vector_set (f, i, (Theo - hits[i])/sqrt(Theo)); // Residual corresponding to Pearson's Chi2
      else gsl_vector_set (f, i, 0); // machine precision
    }
  return GSL_SUCCESS;
}


int do_fit (const int steps, double nevents, double mean, double std, struct data scan, double results[4])
{
  // === Initialise ===
  const gsl_multifit_nlinear_type *T = gsl_multifit_nlinear_trust;
  gsl_multifit_nlinear_workspace *w;
  gsl_multifit_nlinear_fdf fdf;
  gsl_multifit_nlinear_parameters fdf_params = gsl_multifit_nlinear_default_parameters();
  const uint8_t p = 3; // number of free parameters

  gsl_vector *f; // initial cost function
  gsl_matrix *J; // Jacobian
  gsl_matrix *covar = gsl_matrix_alloc (p, p);
  double x_init[p] = { nevents, mean, std };
  gsl_vector_view x = gsl_vector_view_array (x_init, p);

  int status, info;
  const double xtol = 1e-8; // tolerances for convergence
  const double gtol = 1e-8;
  const double ftol = 0.0;

  // === Function to be minimized ===
  fdf.f = func;
  fdf.df = NULL; // No algebraic expression
  fdf.fvv = NULL; // Not used
  fdf.n = steps;
  fdf.p = p;
  fdf.params = &scan;

  // === Allocate workspace and initialise solver ===
  w = gsl_multifit_nlinear_alloc (T, &fdf_params, steps, p);
  gsl_multifit_nlinear_init (&x.vector, &fdf, w);

  // === Solve ===
  status = gsl_multifit_nlinear_driver(100, xtol, gtol, ftol, NULL, NULL, &info, w);

  // === Results ===
  J = gsl_multifit_nlinear_jac(w);
  gsl_multifit_nlinear_covar (J, 0.0, covar);

  f = gsl_multifit_nlinear_residual(w);
  double chisq;
  gsl_blas_ddot(f, f, &chisq);

  results[0] = status;
  results[1] = chisq; // Pearson's chi2
  results[2] = gsl_vector_get(w->x, 1); // mean
  results[3] = gsl_vector_get(w->x, 2); // std

  // == free memory ===
  gsl_multifit_nlinear_free (w);
  gsl_matrix_free (covar);

  return 0;
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
/*
  // (0) study individual pixels
  string name_pix = prefix + "_Pixel.csv";
  FILE *file_pix = fopen(name_pix.c_str(), "w");
  int cdac;
  int r = 195;
  int c = 17;
  int i = r*256 + c; // pixel (1,1) = 257
  for (int dac=0; dac<npoints; ++dac) {
    cdac = dac_max - dac*dac_step;
    //fprintf(file_pix, "%04d, %02d\n", cdac, matrix[i*npoints + dac]);
    fprintf(file_pix, "%02d, ", matrix[i*npoints + dac]);
  }
  fclose(file_pix);
*/

  // (1) calculate mean of noise
  double arg_mean[256*256];
  string name_mean = prefix + "_Noise_Mean.csv";
  FILE *file_mean = fopen(name_mean.c_str(), "w");
  uint16_t nevents[256*256]; // sum of number of hits in across complete scan
  uint16_t smax[256*256]; // maximum number of hits in single scan point
  uint16_t nhits = 0; 
  double imean = 0;
  double imax = 0;

  for (int i=0; i<256*256; ++i) {
    imean = 0;
    imax = 0;
    nhits = 0;
    for (int dac=0; dac<npoints; ++dac) {
      imean += matrix[i*npoints + dac] * (dac_max - dac*dac_step);
      nhits += matrix[i*npoints + dac];
      if (matrix[i*npoints + dac] > imax) imax = matrix[i*npoints + dac];
    }
    nevents[i] = nhits;
    smax[i] = imax;
    if (nhits>0) arg_mean[i] = 1.*imean/nhits;
    else arg_mean[i] = 0;
    
    if (i%256==255) fprintf(file_mean, "%06.1f\n", arg_mean[i]);
    else fprintf(file_mean, "%06.1f, ", arg_mean[i]);
  }
  fclose(file_mean);
  
  
  // (2) calculate width of noise
  string name_width = prefix + "_Noise_Width.csv";
  FILE *file_width = fopen(name_width.c_str(), "w");
  double arg_std[256*256];
  double iwidth = 0;
  
  for (int i=0; i<256*256; ++i) {
    iwidth = 0;
    for (int dac=0; dac<npoints; ++dac) {
      iwidth += matrix[i*npoints + dac] * pow(dac_max - dac*dac_step - arg_mean[i],2);
    }
    if (nevents[i]>0) arg_std[i] = sqrt(1.*iwidth/(nevents[i]-1));
    else arg_std[i] = 0;
      
    if (i%256==255) fprintf(file_width, "%05.2f\n", arg_std[i]);
    else fprintf(file_width, "%05.2f, ", arg_std[i]);
  }
  fclose(file_width);
/*
  // (3) Fit truncated Gaussian
  // Clean up data: 
  // (1) ignore pixels with 0 or 1 hit over whole scan
  // (2) ignore pixels with a max hit less than meanpeak
  // (3) ignore scan points with initial chi2 contribution > chisq_cut
  // This will eliminate:
  // - pixels with no real noise peak response, more continuous noise
  // - scan points outside noise peak with random 1 hit
  // - scan points inside noise peak with random 63 hits

  double minpeak = 10;
  double chisq_cut = 10;
  double theo;
  double chisq;

  double results[4];
  int nfit;
  double x_dac[npoints];
  double y_hits[npoints];

  //string name_fchi = prefix + "_Noise_Fit_chi2.csv";
  //FILE *file_fchi = fopen(name_fchi.c_str(), "w");
  string name_fmean = prefix + "_Noise_Fit_Mean.csv";
  FILE *file_fmean = fopen(name_fmean.c_str(), "w");
  string name_fstd = prefix + "_Noise_Fit_Width.csv";
  FILE *file_fstd = fopen(name_fstd.c_str(), "w");

  //int bad = 0;
  for (int i=0; i<256*256; ++i) {
    nfit = 0;

    //if (nevents[i]>1 && arg_std[i]>1) {
    if (smax[i]>minpeak && arg_std[i]>1) {
      for (int dac=0; dac<npoints; ++dac) {
        x_dac[dac] = 0;
        y_hits[dac] = 0;

        theo = dac_step * nevents[i]/(arg_std[i] * sqrt(4 * asin(1))) * exp( -0.5 * pow((dac_max - dac*dac_step - arg_mean[i])/arg_std[i], 2));
        chisq = pow( matrix[i*npoints + dac] - theo, 2)/theo;
        if (chisq < chisq_cut) {
          x_dac[nfit] = (dac_max - dac*dac_step);
          y_hits[nfit] = matrix[i*npoints + dac];
          nfit++;
        }
      }

      struct data scan = { nfit, x_dac, y_hits };
      do_fit(nfit, dac_step * nevents[i], arg_mean[i], arg_std[i], scan, results);
      // The fit does not always converge, but this is a good indication the pixel is not behaving properly
      // - Scans with order 3 hits in every scan point, but no noise peak
      // - Scans with too few non-zero scan points to determine noise peak properly
      if (results[0]>1) {
        results[1] = -2;
        results[2] = 0;
        results[3] = 0;
      }

      //if (results[3]>24 && results[3]<25 && bad<10) {
      //  fprintf(stderr, "\n\n%d\n", i);
      //  for (int dac=0; dac<npoints; ++dac) {
      //    fprintf(stderr, "%02d, ", matrix[i*npoints + dac]);
      //  }
      //  bad++;
      //}

    } else {
      // No response from pixel
      results[1] = -1;
      results[2] = 0;
      results[3] = 0;
    }
    //if (results[0]!=0) fprintf(stderr, "%d %g\n", i, results[0]);

    //if (i%256==255) fprintf(file_fchi, "%05.2f\n", results[1]);
    //else fprintf(file_fchi, "%05.2f, ", results[1]);

    if (i%256==255) fprintf(file_fmean, "%04.1f\n", results[2]);
    else fprintf(file_fmean, "%04.1f, ", results[2]);

    if (i%256==255) fprintf(file_fstd, "%05.2f\n", results[3]);
    else fprintf(file_fstd, "%05.2f, ", results[3]);
  }
  //fclose(file_fchi);
  fclose(file_fmean);
  fclose(file_fstd);
*/
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
  char next_line[2500];
  vector<string> next_vals;
  int id;

  FILE *file_merge;
  float matrix_merge[256*256];
  const int nMerge = 2;
  string who[nMerge] = {"Noise_Mean", "Noise_Width"};
  string formA[nMerge] = {"%06.1f\n", "%05.2f\n"};
  string formB[nMerge] = {"%06.1f, ", "%05.2f, "};

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
        fgets(next_line, 2500, next_file);
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
