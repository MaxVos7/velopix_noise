CC = g++

all: dim_equalisation dim_decoding

dim_equalisation:	dim_equalisation.cxx
	$(CC) -g -std=c++11 -Wall -Wextra dim_equalisation.cxx -o dim_equalisation

dim_decoding_with_gsl:	dim_decoding_with_gsl.cxx
	$(CC) -g -std=c++11 -Wall -Wextra dim_decoding.cxx -o dim_decoding -lgsl -lgslcblas

dim_decoding:	dim_decoding.cxx
	$(CC) -g -std=c++11 -Wall -Wextra dim_decoding.cxx -o dim_decoding