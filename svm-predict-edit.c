/* Updated libsvm svm-predict to support open set recognition based on

   W. J. Scheirer, A.  Rocha, A. Sapkota, T. E. Boult, "Toward Open Set Recognition," IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 35, no. 7, pp. 1757-1772, July, 2013    

@article{10.1109/TPAMI.2012.256,
author = {W. J. Scheirer and A. Rocha and A. Sapkota and T. E. Boult},
title = {Toward Open Set Recognition},
journal ={IEEE Transactions on Pattern Analysis and Machine Intelligence},
volume = {35},
number = {7},
issn = {0162-8828},
year = {2013},
pages = {1757-1772},
doi = {http://doi.ieeecomputersociety.org/10.1109/TPAMI.2012.256},
}


If you use any of the open set functions please cite appropriately.

There are also extensions using libMR which will be described in other
publications and should also cite based on libMR licensing if that is used as well.


These open set extensions to libSVM are subject to the following

Copyright (c) 2010-2013  Regents of the University of Colorado and Securics Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy of this
software and associated documentation files (the "Software"), to deal in the Software
without restriction, including without limitation the rights to use, copy, modify, merge,
publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons
to whom the Software is furnished to do so, subject to the following 3 conditions:

1) The above copyright notice and this permission notice shall be included in all
source code copies or substantial portions of the Software.

2) All documentation and/or advertising materials mentioning features or use of
this software must display the following acknowledgment:

      This product includes software developed in part at
      the University of Colorado Colorado Springs and Securics Inc.

3) Neither the name of Regents of the University of Colorado  and Securics Inc.  nor
 the names of its contributors may be used to endorse or promote products
 derived from this software without specific prior written permission.


THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


*/ 


#include <stdio.h>
#include <ctype.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <math.h>
#include <glob.h>

#include "svm.h"
int print_null(const char *s,...) {return 0;}

static int (*info)(const char *fmt,...) = &printf;

struct svm_node *x;
int max_nr_attr = 64;

struct svm_model* model;
int predict_probability=0;
double min_threshold = 0, max_threshold = 0;
bool min_set = false, max_set = false;
bool verbose=true;
int debug_level=0;

static char *line = NULL;
static int max_line_len;

//Open set stuff
bool open_set = false;
int nr_classes = 0;
double *lbl;

//score/vote output
bool output_scores = false;
bool output_total_scores = false;
bool output_votes = false;

#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

static char* readline(FILE *input)
{
	int len;
	
	if(fgets(line,max_line_len,input) == NULL)
		return NULL;

	while(strrchr(line,'\n') == NULL)
	{
		max_line_len *= 2;
		line = (char *) realloc(line,(ulong)max_line_len);
		len = (int) strlen(line);
		if(fgets(line+len,max_line_len-len,input) == NULL)
			break;
	}
	return line;
}

void exit_input_error(int line_num)
{
	fprintf(stderr,"Wrong input format at line %d\n", line_num);
	exit(1);
}

void predict(FILE *input, FILE *output)
{
	int correct = 0;
	int total = 0;
	double error = 0;
	double sump = 0, sumt = 0, sumpp = 0, sumtt = 0, sumpt = 0;

	int svm_type = svm_get_svm_type(model);
	int nr_class=svm_get_nr_class(model);
        int ktype=model->param.kernel_type;
	printf("\nSVM Type %d, Kernel Type %d, Num Classes %d \n" svm_type,ktype,nr_class );

	double *prob_estimates=NULL;
	int j;

	if(predict_probability && !open_set)
	{
		if (svm_type==NU_SVR || svm_type==EPSILON_SVR)
			info("Prob. model for test data: target value = predicted value + z,\nz: Laplace distribution e^(-|z|/sigma)/(2sigma),sigma=%g\n",svm_get_svr_probability(model));
		else
		{
			int *labels=(int *) malloc(nr_class*sizeof(int));
			svm_get_labels(model,labels);
			prob_estimates = (double *) malloc(nr_class*sizeof(double));
			fprintf(output,"labels");		
			for(j=0;j<nr_class;j++)
				fprintf(output," %d",labels[j]);
			fprintf(output,"\n");
			free(labels);
		}
	}

	max_line_len = 1024;
	line = (char *)malloc(max_line_len*sizeof(char));
	while(readline(input) != NULL)
	{
		int i = 0;
		double target_label, predict_label = 0;
		char *idx, *val, *label, *endptr;
		int inst_max_index = -1; // strtol gives 0 if wrong format, and precomputed kernel has <index> start from 0

		label = strtok(line," \t\n");
		if(label == NULL) // empty line
			exit_input_error(total+1);

		target_label = strtod(label,&endptr);
		if(endptr == label || *endptr != '\0')
			exit_input_error(total+1);

		while(1)
		{
			if(i>=max_nr_attr-1)	// need one more for index = -1
			{
				max_nr_attr *= 2;
				x = (struct svm_node *) realloc(x,max_nr_attr*sizeof(struct svm_node));
			}

			idx = strtok(NULL,":");
			val = strtok(NULL," \t");

			if(val == NULL)
				break;
			errno = 0;
			x[i].index = (int) strtol(idx,&endptr,10);
			if(endptr == idx || errno != 0 || *endptr != '\0' || x[i].index <= inst_max_index)
				exit_input_error(total+1);
			else
				inst_max_index = x[i].index;

			errno = 0;
			x[i].value = strtod(val,&endptr);
			if(endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr)))
				exit_input_error(total+1);

			++i;
		}
		x[i].index = -1;

		if (predict_probability && (svm_type==C_SVC || svm_type==NU_SVC))
		{
			predict_label = svm_predict_probability(model,x,prob_estimates);
			fprintf(output,"%g",predict_label);
			for(j=0;j<nr_class;j++)
				fprintf(output," %g",prob_estimates[j]);
			fprintf(output,"\n");
		}
		else
		{
			predict_label = svm_predict(model,x);
			fprintf(output,"%g\n",predict_label);
		}

		if(predict_label == target_label)
			++correct;
		error += (predict_label-target_label)*(predict_label-target_label);
		sump += predict_label;
		sumt += target_label;
		sumpp += predict_label*predict_label;
		sumtt += target_label*target_label;
		sumpt += predict_label*target_label;
		++total;
	}

	if (svm_type==NU_SVR || svm_type==EPSILON_SVR )
	{
		info("Mean squared error = %g (regression)\n",error/total);
		info("Squared correlation coefficient = %g (regression)\n",
			((total*sumpt-sump*sumt)*(total*sumpt-sump*sumt))/
			((total*sumpp-sump*sump)*(total*sumtt-sumt*sumt))
			);
	}
	else
	{
	if(predict_probability)
		free(prob_estimates);
}

void exit_with_help()
{
	printf(
	"Usage: svm-predict [options] test_file model_file output_file\n"
	"options:\n"
	"-b probability_estimates: whether to predict probability estimates, 0 or 1 (default 0); for one-class SVM only 0 is supported\n"
	"-q : quiet mode (no outputs)\n"
        "-o : this is an open set problem. this will look for model files with names of the form <model_file>.<class>\n"
        "-V : for more verbose output\n"
        "-s : output scores in bin format(1-2, 1-3, 1-4, 2-3) to outputfile (cannot be combined with -v or -t)\n"
        "-t : output totaled scores 1-2+1-2+1-4=1 etc to output file (cannot be combined with -s or -v)\n"
        "-v : output votes to outputfile(cannot be combined with -s or -t)\n"
	);
	exit(1);
}

int main(int argc, char **argv)
{
	FILE *input, *output;
	int i;
        double openset_min_probability=.001;
	// parse options
	for(i=1;i<argc;i++)
	{
		if(argv[i][0] != '-') break;
		++i;
		switch(argv[i-1][1])
		{
			case 'b':
				predict_probability = atoi(argv[i]);
				break;
			case 'q':
				info = &print_null;
				i--;
				break;
                        case 'P':
                                openset_min_probability = atof(argv[i]);
                                break;
                        case 'o':
                                open_set = true;
                                break;
                        case 'V':
                                verbose = true;
                                break;
                        case 's':
                                output_scores = true;
                                break; 
                        case 'a':
                                output_scores = true;
                                output_votes = true;
                                output_total_scores = true;
                                break; 
                        case 't':
                                output_total_scores = true;
                                break; 
                        case 'v':
                                output_votes = true;
                                break; 
			default:
				fprintf(stderr,"Unknown option: -%c\n", argv[i-1][1]);
				exit_with_help();
		}
	}
	if(i>argc-2)
		exit_with_help();

	input = fopen(argv[i],"r");
	if(input == NULL)
	{
		fprintf(stderr,"can't open input file %s\n",argv[i]);
		exit(1);
	}

	output = fopen(argv[i+2],"w");
	if(output == NULL)
	{
		fprintf(stderr,"can't open output file %s\n",argv[i+2]);
		exit(1);
	}


        if((model=svm_load_model(argv[i+1]))==0)
          {
            fprintf(stderr,"can't open model file %s\n",argv[i+1]);
            exit(1);
          } 
        model->param.openset_min_probability = openset_min_probability;
        if(model && (model->param.svm_type == OPENSET_OC || model->param.svm_type == OPENSET_BIN || model->param.svm_type == OPENSET_PAIR ||model->param.svm_type == ONE_VS_REST_PIESVM )) open_set=true;

	x = (struct svm_node *) malloc(max_nr_attr*sizeof(struct svm_node));
	if(predict_probability && !open_set)
	{
		if(svm_check_probability_model(model)==0)
		{
			fprintf(stderr,"Model does not support probabiliy estimates\n");
			exit(1);
		}
	}
	else if (!open_set)
	{
		if(svm_check_probability_model(model)!=0)
			info("Model supports probability estimates, but disabled in prediction.\n");
	}

	predict(input,output);

        svm_free_and_destroy_model(&model);
	free(x);
	free(line);
	fclose(input);
	fclose(output);
	return 0;
}
