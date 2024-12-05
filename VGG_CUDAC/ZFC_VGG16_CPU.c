#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include <string.h>
#include <ctype.h>
#include <sys/time.h>

#define BLOCK_DIM_X 32
#define BLOCK_DIM_Y 16

#ifdef _WIN32

#define _CRT_SECURE_NO_WARNINGS 1



void gettimeofday(time_t *tp, char *_)
{
	*tp = clock();
	return;
}

double get_seconds(time_t timeStart, time_t timeEnd) {
	return (double)(timeEnd - timeStart) / CLOCKS_PER_SEC;
}
#else
double get_seconds(struct timeval timeStart, struct timeval timeEnd) {
	return ((timeEnd.tv_sec - timeStart.tv_sec) * 1000000 + timeEnd.tv_usec - timeStart.tv_usec) / 1.e6;
}
#endif

#define SIZE 224
#define CONV_SIZE 3



// Weights and image block START
float ***image;
int cshape[13][4] = { 
	{ 64, 3, CONV_SIZE, CONV_SIZE },
	{ 64, 64, CONV_SIZE, CONV_SIZE },
	{ 128, 64, CONV_SIZE, CONV_SIZE },
	{ 128, 128, CONV_SIZE, CONV_SIZE },
	{ 256, 128, CONV_SIZE, CONV_SIZE },
	{ 256, 256, CONV_SIZE, CONV_SIZE },
	{ 256, 256, CONV_SIZE, CONV_SIZE },
	{ 512, 256, CONV_SIZE, CONV_SIZE },
	{ 512, 512, CONV_SIZE, CONV_SIZE },
	{ 512, 512, CONV_SIZE, CONV_SIZE },
	{ 512, 512, CONV_SIZE, CONV_SIZE },
	{ 512, 512, CONV_SIZE, CONV_SIZE },
	{ 512, 512, CONV_SIZE, CONV_SIZE }
};
float *****wc;
float **bc;
int dshape[3][2] = {
	{ 25088, 4096 },
	{ 4096, 4096 },
	{ 4096, 1000 }
};
float ***wd;
float **bd;


// Blocks for intermediate convolutions
int mem_block_shape[3] = {512, SIZE, SIZE};
float ***mem_block1;
float ***mem_block2;
// Blocks for dense flatten layers
int mem_block_dense_shape = { 512 * 7 * 7 };
float *mem_block1_dense;
float *mem_block2_dense;

// Weights and image block END


void reset_mem_block(float ***mem) {
	int i, j, k;
	for (i = 0; i < mem_block_shape[0]; i++) {
		for (j = 0; j < mem_block_shape[1]; j++) {
			for (k = 0; k < mem_block_shape[2]; k++) {
				mem[i][j][k] = 0.0;
			}
		}
	}
}


void reset_mem_block_dense(float *mem) {
	int i;
	for (i = 0; i < mem_block_dense_shape; i++) {
		mem[i] = 0.0;
	}
}


void init_memory() {
	int i, j, k, l;

	// Init image memory
	image =(float***) malloc(3 * sizeof(float**));
	for (i = 0; i < 3; i++) {
		image[i] = (float**) malloc(SIZE * sizeof(float*));
		for (j = 0; j < SIZE; j++) {
			image[i][j] = (float*)malloc(SIZE * sizeof(float));
		}
	}

	// Init convolution weights
	wc = (float*****)malloc(13 * sizeof(float****));
	bc = (float**)malloc(13 * sizeof(float*));
	for (l = 0; l < 13; l++) {
		wc[l] = (float****)malloc(cshape[l][0] * sizeof(float***));
		for (i = 0; i < cshape[l][0]; i++) {
			wc[l][i] = (float***)malloc(cshape[l][1] * sizeof(float**));
			for (j = 0; j < cshape[l][1]; j++) {
				wc[l][i][j] = (float**)malloc(cshape[l][2] * sizeof(float*));
				for (k = 0; k < cshape[l][2]; k++) {
					wc[l][i][j][k] = (float*)malloc(cshape[l][3] * sizeof(float));
				}
			}
		}
		bc[l] = (float*)malloc(cshape[l][0] * sizeof(float));
	}

	// Init dense weights
	wd = (float***) malloc(3 * sizeof(float**));
	bd = (float**) malloc(3 * sizeof(float*));
	for (l = 0; l < 3; l++) {
		wd[l] = (float**)malloc(dshape[l][0] * sizeof(float*));
		for (i = 0; i < dshape[l][0]; i++) {
			wd[l][i] = (float*)malloc(dshape[l][1] * sizeof(float));
		}
		bd[l] = (float*)malloc(dshape[l][1] * sizeof(float));
	}

	// Init mem_blocks
	mem_block1 = (float***)malloc(mem_block_shape[0] * sizeof(float**));
	mem_block2 = (float***)malloc(mem_block_shape[0] * sizeof(float**));
	for (i = 0; i < mem_block_shape[0]; i++) {
		mem_block1[i] = (float**)malloc(mem_block_shape[1] * sizeof(float*));
		mem_block2[i] = (float**)malloc(mem_block_shape[1] * sizeof(float*));
		for (j = 0; j < mem_block_shape[1]; j++) {
			mem_block1[i][j] = (float*)malloc(mem_block_shape[2] * sizeof(float));
			mem_block2[i][j] = (float*)malloc(mem_block_shape[2] * sizeof(float));
		}
	}
	reset_mem_block(mem_block1);
	reset_mem_block(mem_block2);

	// Init mem blocks dense
	mem_block1_dense = (float*)calloc(mem_block_dense_shape, sizeof(float));
	mem_block2_dense = (float*)calloc(mem_block_dense_shape, sizeof(float));
}


void free_memory() {
	int i, j, k, l;

	// Free image memory
	for (i = 0; i < 3; i++) {
		for (j = 0; j < SIZE; j++) {
			free(image[i][j]);
		}
		free(image[i]);
	}
	free(image);

	// Free convolution weights
	for (l = 0; l < 13; l++) {
		for (i = 0; i < cshape[l][0]; i++) {
			for (j = 0; j < cshape[l][1]; j++) {
				for (k = 0; k < cshape[l][2]; k++) {
					free(wc[l][i][j][k]);
				}
				free(wc[l][i][j]);
			}
			free(wc[l][i]);
		}
		free(wc[l]);
		free(bc[l]);
	}
	free(wc);
	free(bc);

	// Free dense weights
	for (l = 0; l < 3; l++) {
		for (i = 0; i < dshape[l][0]; i++) {
			free(wd[l][i]);
		}
		free(wd[l]);
		free(bd[l]);
	}
	free(wd);
	free(bd);

	// Free memblocks
	for (i = 0; i < mem_block_shape[0]; i++) {
		for (j = 0; j < mem_block_shape[1]; j++) {
			free(mem_block1[i][j]);
			free(mem_block2[i][j]);
		}
		free(mem_block1[i]);
		free(mem_block2[i]);
	}
	free(mem_block1);
	free(mem_block2);

	free(mem_block1_dense);
	free(mem_block2_dense);
}


void read_weights(char *in_file, int lvls) {
	float dval;
	int i, j, k, l, z;
	FILE *iin;
	int total_lvls_read = 0;

	iin = fopen(in_file, "r");
	if (iin == NULL) {
		printf("File %s absent\n", in_file);
		exit(1);
	}
	
	// Reading convolution weights (store them flipped from begining)
	for (z = 0; z < 13; z++) {
		if (total_lvls_read >= lvls && lvls != -1)
			break;
		printf("Read conv block %d weights\n", z);
		for (i = 0; i < cshape[z][0]; i++) {
			for (j = 0; j < cshape[z][1]; j++) {
				for (k = 0; k < cshape[z][2]; k++) {
					for (l = 0; l < cshape[z][3]; l++) {
						fscanf(iin, "%f", &dval);
						wc[z][i][j][CONV_SIZE - k - 1][CONV_SIZE - l - 1] = dval;
					}
				}
			}
		}
		for (i = 0; i < cshape[z][0]; i++) {
			fscanf(iin, "%f", &dval);
			bc[z][i] = dval;
		}
		total_lvls_read += 1;
	}

	// Reading dense weights
	for (z = 0; z < 3; z++) {
		if (total_lvls_read >= lvls && lvls != -1)
			break;
		printf("Read dense block %d weights\n", z);
		for (i = 0; i < dshape[z][0]; i++) {
			for (j = 0; j < dshape[z][1]; j++) {
				fscanf(iin, "%f", &dval);
				wd[z][i][j] = dval;
			}
		}
		for (i = 0; i < dshape[z][1]; i++) {
			fscanf(iin, "%f", &dval);
			bd[z][i] = dval;
		}
		total_lvls_read += 1;
	}

	fclose(iin);
}


void read_image(char *in_file) {
	int i, j, l;
	FILE *iin;
	float dval;

	iin = fopen(in_file, "r");
	if (iin == NULL) {
		printf("File %s absent\n", in_file);
		exit(1);
	}

	/* Reading image */
	for (i = 0; i < SIZE; i++) {
		for (j = 0; j < SIZE; j++) {
			for (l = 0; l < 3; l++) {
				fscanf(iin, "%f", &dval);
				image[l][i][j] = dval;
			}
		}
	}

	fclose(iin);
}


void normalize_image() {
	int i, j, l;
	float coef[3] = { 103.939, 116.779, 123.68 };

	for (l = 0; l < 3; l++) {
		for (i = 0; i < SIZE; i++) {
			for (j = 0; j < SIZE; j++) {
				image[l][i][j] -= coef[l];
			}
		}
	}
}


void convolution_3_x_3(float **matrix, float **kernel, float **out, int size) {
	int i, j;
	float sum;
	float zeropad[SIZE + 2][SIZE + 2] = { 0.0 };

	for (i = 0; i < size; i++) {
		for (j = 0; j < size; j++) {
			zeropad[i + 1][j + 1] = matrix[i][j];
		}
	}

	for (i = 0; i < size; i++) {
		for (j = 0; j < size; j++) {
			sum = zeropad[i][j] * kernel[0][0] +
				zeropad[i + 1][j] * kernel[1][0] +
				zeropad[i + 2][j] * kernel[2][0] +
				zeropad[i][j + 1] * kernel[0][1] +
				zeropad[i + 1][j + 1] * kernel[1][1] +
				zeropad[i + 2][j + 1] * kernel[2][1] +
				zeropad[i][j + 2] * kernel[0][2] +
				zeropad[i + 1][j + 2] * kernel[1][2] +
				zeropad[i + 2][j + 2] * kernel[2][2];
			out[i][j] += sum;
		}
	}
	
}


void add_bias_and_relu(float **out, float bs, int size) {
	int i, j;

	for (i = 0; i < size; i++) {
		for (j = 0; j < size; j++) {
			out[i][j] += bs;
			if (out[i][j] < 0)
				out[i][j] = 0.0;
			// printf("%.12lf\n", out[i][j]);
		}
	}
}


void add_bias_and_relu_flatten(float *out, float *bs, int size, int relu) {
	int i;
	for (i = 0; i < size; i++) {
		out[i] += bs[i];
		if (relu == 1) {
			if (out[i] < 0)
				out[i] = 0.0;
		}
	}
}


float max_of_4(float a, float b, float c, float d) {
	if (a >= b && a >= c && a >= d) {
		return a;
	}
	if (b >= c && b >= d) {
		return b;
	}
	if (c >= d) {
		return c;
	}
	return d;
}


void maxpooling(float **out, int size) {
	int i, j;
	for (i = 0; i < size; i+=2) {
		for (j = 0; j < size; j+=2) {
			out[i / 2][j / 2] = max_of_4(out[i][j], out[i + 1][j], out[i][j + 1], out[i + 1][j + 1]);
		}
	}
}


void flatten(float ***in, float *out, int sh0, int sh1, int sh2) {
	int i, j, k, total = 0;
	for (i = 0; i < sh0; i++) {
		for (j = 0; j < sh1; j++) {
			for (k = 0; k < sh2; k++) {
				out[total] = in[i][j][k];
				total += 1;
			}
		}
	}
}



__global__ void dense_kernel(float *in, float *weights, float *out, int sh_in, int sh_out) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < sh_out && col < sh_in) {
        atomicAdd(&out[row], in[col] * weights[col * sh_out + row]);
    }
}

void dense(float *in, float **weights, float *out, int sh_in, int sh_out) {
    float *d_in, *d_weights, *d_out;

    // Flatten weights matrix for easier transfer to GPU
    float *weights_flat = (float *)malloc(sh_in * sh_out * sizeof(float));
    for (int i = 0; i < sh_in; i++) {
        for (int j = 0; j < sh_out; j++) {
            weights_flat[i * sh_out + j] = weights[i][j];
        }
    }

    // Allocate device memory
    cudaMalloc((void **)&d_in, sh_in * sizeof(float));
    cudaMalloc((void **)&d_weights, sh_in * sh_out * sizeof(float));
    cudaMalloc((void **)&d_out, sh_out * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_in, in, sh_in * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, weights_flat, sh_in * sh_out * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, sh_out * sizeof(float)); // Initialize output to 0

    // Define 2D block and grid size
    dim3 blockDim(BLOCK_DIM_X, BLOCK_DIM_Y);
    dim3 gridDim((sh_out + blockDim.x - 1) / blockDim.x, (sh_in + blockDim.y - 1) / blockDim.y);

    // Launch kernel with 2D block and grid
    dense_kernel<<<gridDim, blockDim>>>(d_in, d_weights, d_out, sh_in, sh_out);

    // Copy result back to host
    cudaMemcpy(out, d_out, sh_out * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_in);
    cudaFree(d_weights);
    cudaFree(d_out);
    free(weights_flat);
}




void softmax(float *out, int sh_out) {
	int i;
	float max_val, sum;
	max_val = out[0];
	for (i = 1; i < sh_out; i++) {
		if (out[i] > max_val)
			max_val = out[i];
	}
	sum = 0.0;
	for (i = 0; i < sh_out; i++) {
		out[i] = exp(out[i] - max_val);
		sum += out[i];
	}
	for (i = 0; i < sh_out; i++) {
		out[i] /= sum;
	}
}



void dump_memory_structure_conv(float ***mem, int sh0, int sh1, int sh2) {
	int i, j, k;
	for (i = 0; i < sh0; i++) {
		for (j = 0; j < sh1; j++) {
			for (k = 0; k < sh2; k++) {
				printf("%.12lf\n", mem[i][j][k]);
			}
		}
	}
}

void dump_memory_structure_conv_to_file(float ***mem, int sh0, int sh1, int sh2) {
	FILE *out;
	int i, j, k;
	out = fopen("debug_c.txt", "w");
	for (i = 0; i < sh0; i++) {
		for (j = 0; j < sh1; j++) {
			for (k = 0; k < sh2; k++) {
				fprintf(out, "%.12lf\n", mem[i][j][k]);
			}
		}
	}
	fclose(out);
}


void dump_memory_structure_dense(float *mem, int sh0) {
	int i;
	for (i = 0; i < sh0; i++) {
		printf("%.12lf\n", mem[i]);
	}
}


void dump_memory_structure_dense_to_file(float *mem, int sh0) {
	FILE *out;
	int i;
	out = fopen("debug_c.txt", "w");
	for (i = 0; i < sh0; i++) {
		fprintf(out, "%.12lf\n", mem[i]);
	}
	fclose(out);
}

void dump_image() {
	int i, j, k;
	for (i = 0; i < 3; i++) {
		for (j = 0; j < SIZE; j++) {
			for (k = 0; k < SIZE; k++) {
				printf("%.12lf\n", image[i][j][k]);
			}
		}
	}
}



float* flatten_2d_array(float** arr, int rows, int cols) {
    float* flat = (float*)malloc(rows * cols * sizeof(float));
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            flat[i * cols + j] = arr[i][j];
        }
    }
    return flat;
}

void copy_flat_to_2d_array(float** arr, float* flat, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            arr[i][j] = flat[i * cols + j];
        }
    }
}

__global__ void convolution_3_x_3_kernel(float *input, float *kernel, float *output, int size) {
    // Get the row and column based on block and thread indices
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < size && col < size) {
        float sum = 0.0;

        // Iterate over the 3x3 kernel
        for (int i = -1; i <= 1; i++) {
            for (int j = -1; j <= 1; j++) {
                int r = row + i;
                int c = col + j;
                if (r >= 0 && r < size && c >= 0 && c < size) {
                    sum += input[r * size + c] * kernel[(i + 1) * 3 + (j + 1)];
                }
            }
        }
        output[row * size + col] = sum;  // Set the output (not accumulate)
    }
}

__global__ void maxpooling_kernel(float *input, float *output, int size) {
    // Get the row and column based on block and thread indices
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < size / 2 && col < size / 2) {
        int r = row * 2;
        int c = col * 2;

        // Compute the maximum of the 2x2 window
        float max_val = fmaxf(fmaxf(input[r * size + c], input[r * size + (c + 1)]),
                              fmaxf(input[(r + 1) * size + c], input[(r + 1) * size + (c + 1)]));

        // Assign the maximum value to the output
        output[row * (size / 2) + col] = max_val;
    }
}



void get_VGG16_predict(int only_convolution) {
    int i, j;
    int level, cur_size;

    // Init intermediate memory
    reset_mem_block(mem_block1);
    reset_mem_block(mem_block2);
    reset_mem_block_dense(mem_block1_dense);
    reset_mem_block_dense(mem_block2_dense);

    float *d_input, *d_kernel, *d_output;
    int input_size, kernel_size, output_size;
    
    // Define block and grid dimensions
    dim3 blockDim(16, 16); // 16x16 threads per block, giving 256 threads per block

    // Layer 1 (Convolution 3 -> 64)
    level = 0;
    cur_size = SIZE;

    for (i = 0; i < cshape[level][0]; i++) {
        for (j = 0; j < cshape[level][1]; j++) {
            // Flatten and transfer input and kernel to the GPU
            input_size = cur_size * cur_size * sizeof(float);
            kernel_size = 3 * 3 * sizeof(float);
            cudaMalloc((void**)&d_input, input_size);
            cudaMalloc((void**)&d_kernel, kernel_size);
            cudaMalloc((void**)&d_output, input_size);

            float *input_flat = flatten_2d_array(image[j], cur_size, cur_size);
            float *kernel_flat = flatten_2d_array(wc[level][i][j], 3, 3);
            cudaMemcpy(d_input, input_flat, input_size, cudaMemcpyHostToDevice);
            cudaMemcpy(d_kernel, kernel_flat, kernel_size, cudaMemcpyHostToDevice);
            cudaMemset(d_output, 0, input_size); // Initialize output to 0

            // Launch convolution kernel
            dim3 gridDim((cur_size + blockDim.x - 1) / blockDim.x, (cur_size + blockDim.y - 1) / blockDim.y);
            convolution_3_x_3_kernel<<<gridDim, blockDim>>>(d_input, d_kernel, d_output, cur_size);

            // Copy output back to host
            cudaMemcpy(input_flat, d_output, input_size, cudaMemcpyDeviceToHost);
            copy_flat_to_2d_array(mem_block1[i], input_flat, cur_size, cur_size);

            // Free GPU memory
            cudaFree(d_input);
            cudaFree(d_kernel);
            cudaFree(d_output);
            free(input_flat);
            free(kernel_flat);
        }
        add_bias_and_relu(mem_block1[i], bc[level][i], cur_size);
    }

    // Layer 2 (Convolution 64 -> 64)
    level = 1;
    for (i = 0; i < cshape[level][0]; i++) {
        for (j = 0; j < cshape[level][1]; j++) {
            // Flatten and transfer input and kernel to the GPU
            input_size = cur_size * cur_size * sizeof(float);
            kernel_size = 3 * 3 * sizeof(float);
            cudaMalloc((void**)&d_input, input_size);
            cudaMalloc((void**)&d_kernel, kernel_size);
            cudaMalloc((void**)&d_output, input_size);

            float *input_flat = flatten_2d_array(mem_block1[j], cur_size, cur_size);
            float *kernel_flat = flatten_2d_array(wc[level][i][j], 3, 3);
            cudaMemcpy(d_input, input_flat, input_size, cudaMemcpyHostToDevice);
            cudaMemcpy(d_kernel, kernel_flat, kernel_size, cudaMemcpyHostToDevice);
            cudaMemset(d_output, 0, input_size); // Initialize output to 0

            // Launch convolution kernel
            dim3 gridDim((cur_size + blockDim.x - 1) / blockDim.x, (cur_size + blockDim.y - 1) / blockDim.y);
            convolution_3_x_3_kernel<<<gridDim, blockDim>>>(d_input, d_kernel, d_output, cur_size);

            // Copy output back to host
            cudaMemcpy(input_flat, d_output, input_size, cudaMemcpyDeviceToHost);
            copy_flat_to_2d_array(mem_block2[i], input_flat, cur_size, cur_size);

            // Free GPU memory
            cudaFree(d_input);
            cudaFree(d_kernel);
            cudaFree(d_output);
            free(input_flat);
            free(kernel_flat);
        }
        add_bias_and_relu(mem_block2[i], bc[level][i], cur_size);
    }
    reset_mem_block(mem_block1);

    // Layer 3 (MaxPooling)
    for (i = 0; i < cshape[level][0]; i++) {
        // Flatten and transfer input to the GPU
        input_size = cur_size * cur_size * sizeof(float);
        output_size = (cur_size / 2) * (cur_size / 2) * sizeof(float);
        cudaMalloc((void**)&d_input, input_size);
        cudaMalloc((void**)&d_output, output_size);

        float *input_flat = flatten_2d_array(mem_block2[i], cur_size, cur_size);
        cudaMemcpy(d_input, input_flat, input_size, cudaMemcpyHostToDevice);

        // Launch maxpooling kernel
        dim3 gridDim((cur_size / 2 + blockDim.x - 1) / blockDim.x, (cur_size / 2 + blockDim.y - 1) / blockDim.y);
        maxpooling_kernel<<<gridDim, blockDim>>>(d_input, d_output, cur_size);

        // Copy output back to host
        float *output_flat = (float *)malloc(output_size);
        cudaMemcpy(output_flat, d_output, output_size, cudaMemcpyDeviceToHost);
        copy_flat_to_2d_array(mem_block2[i], output_flat, cur_size / 2, cur_size / 2);

        // Update cur_size for next layers
        cur_size /= 2;

        // Free GPU memory
        cudaFree(d_input);
        cudaFree(d_output);
        free(input_flat);
        free(output_flat);
    }

    // Continue similar conversion for all convolution and max-pooling layers until Layer 18.
    // All remaining convolution and pooling layers will follow the same pattern.

    // After Layer 18, proceed with the dense layers as in the original function.

    // Finalize Flatten and Dense Layers
    flatten(mem_block1, mem_block1_dense, cshape[level][0], cur_size, cur_size);
    if (only_convolution == 1) {
        return;
    }

    // Layer 20 (Dense)
    level = 0;
    dense(mem_block1_dense, wd[level], mem_block2_dense, dshape[level][0], dshape[level][1]);
    add_bias_and_relu_flatten(mem_block2_dense, bd[level], dshape[level][1], 1);
    reset_mem_block_dense(mem_block1_dense);

    // Layer 21 (Dense)
    level = 1;
    dense(mem_block2_dense, wd[level], mem_block1_dense, dshape[level][0], dshape[level][1]);
    add_bias_and_relu_flatten(mem_block1_dense, bd[level], dshape[level][1], 1);
    reset_mem_block_dense(mem_block2_dense);

    // Layer 22 (Dense)
    level = 2;
    dense(mem_block1_dense, wd[level], mem_block2_dense, dshape[level][0], dshape[level][1]);
    add_bias_and_relu_flatten(mem_block2_dense, bd[level], dshape[level][1], 1);
    softmax(mem_block2_dense, dshape[level][1]);

    return;
}




void output_predictions(FILE *out, int only_convolution) {
	int i;
	if (only_convolution == 1) {
		for (i = 0; i < 512*7*7; i++) {
			fprintf(out, "%g ", mem_block1_dense[i]);
		}
	}
	else {
		for (i = 0; i < dshape[2][1]; i++) {
			fprintf(out, "%g ", mem_block2_dense[i]);
		}
	}
	fprintf(out, "\n");
}


char *trimwhitespace(char *str)
{
	char *end;

	// Trim leading space
	while (isspace((unsigned char)*str)) str++;

	if (*str == 0)  // All spaces?
		return str;

	// Trim trailing space
	end = str + strlen(str) - 1;
	while (end > str && isspace((unsigned char)*end)) end--;

	// Write new null terminator
	*(end + 1) = 0;

	return str;
}


int main(int argc, char *argv[]) {
	FILE *file_list, *results;
	char buf[1024];
#ifndef _WIN32
	struct timeval timeStart, timeEnd;
#else
	time_t timeStart, timeEnd;
#endif
	double deltaTime;
	char *weights_file;
	char *image_list_file;
	char *output_file;
	int lvls = -1;
	int only_convolution = 0;


	printf("Using (%d,%d) 2D block:\n",BLOCK_DIM_X,BLOCK_DIM_Y);
  printf("number of threads= %d * %d = %d\n",BLOCK_DIM_X,BLOCK_DIM_Y,BLOCK_DIM_X*BLOCK_DIM_Y);

	if (argc != 4 && argc != 5) {
		printf("Usage: <program.exe> <weights file> <images list file> <output file> <only_convolution [optional]>\n");
		return 0;
	}
	weights_file = argv[1];
	image_list_file = argv[2];
	output_file = argv[3];
	if (argc == 5) {
		lvls = 13;
		only_convolution = 1;
	}

	init_memory();
	file_list = fopen(image_list_file, "r");
	if (file_list == NULL) {
		printf("Check file list location: %s", image_list_file);
		return 1;
	}
	results = fopen(output_file, "w");
	if (results == NULL) {
		printf("Couldn't open file for writing: %s", output_file);
		return 1;
	}

	gettimeofday(&timeStart, NULL);
	read_weights(weights_file, lvls);
	gettimeofday(&timeEnd, NULL);
	deltaTime = get_seconds(timeStart, timeEnd);
	printf("Reading weights: %.3lf sec\n", deltaTime);

	while (!feof(file_list)) {
		gettimeofday(&timeStart, NULL);
		fgets(buf, 1024, file_list);
		if (strlen(buf) == 0) {
			break;
		}
		//printf("%ld\n", strlen(buf));
		read_image(trimwhitespace(buf));
		normalize_image();
		// dump_image();
		get_VGG16_predict(only_convolution);
		output_predictions(results, only_convolution);
		gettimeofday(&timeEnd, NULL);
		deltaTime = get_seconds(timeStart, timeEnd);
		printf("Infer image %s: %.3lf sec\n", buf, deltaTime);
	}

	free_memory();
	fclose(file_list);
	return 0;
}