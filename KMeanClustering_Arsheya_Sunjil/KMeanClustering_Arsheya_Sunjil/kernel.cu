//--------------------------------------------------------------------------------------------------------------------------------------------------
// Final Project : K-Means Sequential and Parallel Clustering (KMeansClustering_Arsheya_Sunjil)
// Implement K-means clustering for a large dataset using parallelization in CUDA.
// Author:  Sunjil Gahatraj, Arsheya Raj
// Date: 3/16/2023
//--------------------------------------------------------------------------------------------------------------------------------------------------
//
//	In this project we implemented the K-means clustering for a large dataset using parallelization in CUDA. This project also contains the 
//	sequential implementation of k-means clustering too.
//  We also used the NVIDIA Nsight Systems profiler to profile our implementation. 
// 
//	Implementation is written in CUDA using Visual Studio 2022. To run the program, copy the “.snl” file to Visual Studio and 
//	build project using “Build” button. Output and inputfile path should be updated to match correct location in your the machine.
//	Update the path for both the implentation of sequential and parallel for the output files.
//	Update the path in the python script to match the location of the centroids csv file to get the plot.
// 
//--------------------------------------------------------------------------------------------------------------------------------------------------

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <time.h>
#include <iostream>     
#include <sstream>      
#include <fstream>      
#include <ctime>     	
#include <chrono>	// for measuring the time

using namespace std::chrono;
using namespace std;

#define D 2 						// Dimension of points
#define K 10						// Number of clusters
#define TPB 256						// Number of threads per block
#define __FLT_MAX__	1000000000.00	// Arbitrary Float Max

//--------------------------------------------Euclidean distance of two 2D points for parallel implementation---------------------------------------
__device__ float euclideandistance(float x1, float y1, float x2, float y2){
	return sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1));
}

//--------------------------------------------KMeans Cluster Assigment by calculating the distances-------------------------------------------------
__global__ void kMeansClusterAssignmentusingdistance_parallel(float* d_datapoints, int* d_clust_assn, float* d_centroids, int N){
	//get the index for this datapoint
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;

	//bounds check for the thread
	if (idx >= N) return;

	//find the closest centroid to this datapoint
	float min_dist = __FLT_MAX__;
	int closest_centroid = -1;

	for (int c = 0; c < K; ++c){
		// data points = [x1, y1,...,xn, yn], centroids = [c1x, c1y,..., ckx, cky]

		float distance = euclideandistance(d_datapoints[2 * idx], d_datapoints[2 * idx + 1], d_centroids[2 * c], d_centroids[2 * c + 1]);

		// Update to the new cluster if it's closer 
		if (distance < min_dist){
			min_dist = distance;        // update the minimum distance to the current distance
			closest_centroid = c;		// update the current closest centroid for the datapoint
		}
	}
	//assign closest cluster id for this datapoint/thread
	d_clust_assn[idx] = closest_centroid;
}

//--------------------------------------------centroid initializtion for parallel implementation----------------------------------------------------
__host__ void centroid_initialization_parallel(float* h_datapoints, float* h_centroids, int N){
	//initalization of the centroids	
	for (int i = 0; i < K; i++) {
		int temp = (N / K);
		int idx_r = rand() % temp;
		h_centroids[2 * i] = h_datapoints[(i * temp + idx_r)];
		h_centroids[2 * i + 1] = h_datapoints[(i * temp + idx_r) + 1];
	}
};

//-----------------------------------Updating the new centroids according to the mean value of all the assigned data points-------------------------
// This code is commented out as it is to be completed as future work. Currently, when we uncomment this code and run it for centroid updation,
// it doesn't produce any output, rather it has only (0,0) as the centroids for the whole solution. This is the optimization of our implementation
// which needs to be implemented.
/*__global__ void kMeansCentroidUpdate_parallel_new(float* data, int* labels, float* centroids, int k, int n, int d){

	extern __shared__ float shared_mem[];

	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	for (int i = threadIdx.x; i < k * d; i += blockDim.x) {
		shared_mem[i] = 0.0f;
	}

	__syncthreads();

	// Calculate sum of all data points belonging to each centroid
	for (int i = tid; i < n; i += blockDim.x * gridDim.x) {
		int label = labels[i];
		for (int j = 0; j < d; j++) {
			atomicAdd(&shared_mem[label * d + j], data[i * d + j]);
		}
	}

	__syncthreads();

	// Update centroids
	for (int i = threadIdx.x; i < k; i += blockDim.x) {
		for (int j = 0; j < d; j++) {
			centroids[i * d + j] = shared_mem[i * d + j] / (float)atomicAdd(&shared_mem[k * d + j], 1);
		}
	}
}
*/

//-----------------------------------Updating the new centroids according to the mean value of all the assigned data points-------------------------
void kMeansCentroidUpdate_parallel(float* h_datapoints, int* h_clust_assn, float* h_centroids, int* h_clust_sizes, int N){

	float cluster_datapoint_sums[2 * K] = { 0 };

	for (int j = 0; j < N; ++j)
	{
		// cluster_id represents a cluster from 1...K
		int cluster_id = h_clust_assn[j];
		cluster_datapoint_sums[2 * cluster_id] += h_datapoints[2 * j];
		cluster_datapoint_sums[2 * cluster_id + 1] += h_datapoints[2 * j + 1];
		h_clust_sizes[cluster_id] += 1;
	}

	//Division by size (arithmetic mean) to compute the actual centroids
	for (int idx = 0; idx < K; idx++) {
		if (h_clust_sizes[idx])
		{
			h_centroids[2 * idx] = cluster_datapoint_sums[2 * idx] / h_clust_sizes[idx];
			h_centroids[2 * idx + 1] = cluster_datapoint_sums[2 * idx + 1] / h_clust_sizes[idx];
		}
	}

}

//--------------------------------------------Euclidean distance of two 2D point for sequential implementation--------------------------------------
float euclideandistance_seq(float x1, float y1, float x2, float y2){
	return sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1));
}

//-----------------------Finding the closest centroid to each of N data-points for each cluster K_seq for sequential implementation-----------------
void kMeansClusterAssignment_seq(float* datapoints, int* clust_assn, float* centroids, int N, int K_seq){
	for (int idx = 0; idx < N; idx++){
		float min_dist = __FLT_MAX__;
		int closest_centroid = -1;

		// distance of one point from datapoints and centroid of each cluster
		for (int c = 0; c < K_seq; ++c){
			// data-points in the format of [x1, y1,...,xn, yn], centroids in the format of [c1x, c1y,..., ckx, cky]
			float distance = euclideandistance_seq(datapoints[2 * idx], datapoints[2 * idx + 1], centroids[2 * c], centroids[2 * c + 1]);

			// update of new cluster if it's closer 
			if (distance < min_dist){
				min_dist = distance;		// update the minimum distance to the current distance which is less than min_dist
				closest_centroid = c;		// Assign the current closest centroid for the datapoint
			}
		}
		// assign the cluster to that point after iteration through all the clusters
		clust_assn[idx] = closest_centroid;
	}
}

//--------------------------------------------centroid initializtion for sequential implementation--------------------------------------------------
void centroid_initialization_seq(float* datapoints, float* centroids, int N, int K_seq) {
	for (int c = 0; c < K_seq; c++) {
		int temp = (N / K_seq);
		int idx_r = rand() % temp;

		// for each cluster choosing randomly the centroid
		centroids[2 * c] = datapoints[(c * temp + idx_r)];
		centroids[2 * c + 1] = datapoints[(c * temp + idx_r) + 1];
	}
};

//-------------Updating the new centroids according to the mean value of all the assigned data points for Sequential implementation-----------------
void kMeansCentroidUpdate_seq(float* datapoints, int* clust_assn, float* centroids, int* clust_sizes, int N, int K_seq){
	float cluster_datapoint_sums[2 * 50] = { 0 };
	for (int idx = 0; idx < N; ++idx)
	{
		// cluster_id represents a number of a cluster from 0 to K
		int cluster_id = clust_assn[idx];

		// summation of both of the coordinates for each cluster
		cluster_datapoint_sums[2 * cluster_id] += datapoints[2 * idx];				// for x coordinates
		cluster_datapoint_sums[2 * cluster_id + 1] += datapoints[2 * idx + 1];      // for y coordinates

		// counting of the total number of data points within each cluster
		clust_sizes[cluster_id] += 1;
	}

	// finding the arithmetic mean to get the current updated centroid within each cluster
	for (int c = 0; c < K_seq; c++) {
		if (clust_sizes[c]){ // this is to avoid division by zero when there is no point in the cluster
			centroids[2 * c] = cluster_datapoint_sums[2 * c] / clust_sizes[c];			// new x coordinate of the updated centroid
			centroids[2 * c + 1] = cluster_datapoint_sums[2 * c + 1] / clust_sizes[c];  // new y coordinate of the updated centroid
		}
	}
}

//---------------------------------------------------------File Reader------------------------------------------------------------------------------
bool Read_from_file(float* h_datapoints, std::string input_file = "D:\School\MasterDegree\CSS535\Final Project\KMeanClustering_Arsheya_Sunjil\KMeanClustering_Arsheya_Sunjil\option1_100_points.txt") {

	//initalize datapoints
	FILE* file = fopen(input_file.c_str(), "r");

	if (file != NULL) {
		int d = 0;
		while (!feof(file))
		{
			float x, y;

			// break if you will not find a pair
			if (fscanf(file, "%f %f", &x, &y) != 2) {
				break;
			}
			h_datapoints[2 * d] = x;
			h_datapoints[2 * d + 1] = y;
			d = d + 1;
		}
		fclose(file);
		return 0;

	}
	else {
		cerr << "Error during opening file \n";
		return -1;
	}
};

//----------------------------------------------Generating the output files in csv format-----------------------------------------------------------
void write2csv(float* points, std::string outfile_name, int size)	// size is the number of points 
{
	std::ofstream outfile;
	outfile.open(outfile_name);
	outfile << "x,y\n";  // name of the columns for csv

	for (int i = 0; i < size; i++) {
		outfile << points[2 * i] << "," << points[2 * i + 1] << "\n";
	}
}

//-----------------------Generating the output file for cluster assignment(c) for the data points (x, y) in csv format------------------------------
void write2csv_clust(float* points, int* clust_assn,
	std::string outfile_name, int size)
{
	std::ofstream outfile;
	outfile.open(outfile_name);
	outfile << "x,y,c\n";  // name of the columns in csv file

	// writing of the coordinates of data-points and their relative cluster in the csv output file
	for (int i = 0; i < size; i++) {
		outfile << points[2 * i] << "," << points[2 * i + 1] << "," << clust_assn[i] << "\n";
	}
}

//-------------------User can define the number of: data points (N), iterations and clusters for sequential implementation--------------------------
void input_user_seq(std::string* infile_name, int* num, int* k, int* iterations){
	cout << "Number (int) of points you want to analyze (100, 1000):\n";
	std::cin >> *num;
	int n = *num;

	switch (n){
	case 100: *infile_name = "option1_100_points.txt";
		break;
	case 1000: *infile_name = "option2_1000_points.txt";
		break;
	default: *infile_name = "option1_100_points.txt";
		cout << "The Dataset with " << n << " points does not exist!\nThe \"option1_100_points.txt\" dataset is chosen instead of default.\n" << endl;
		break;
	}

	cout << "Please, insert number (int) of iterations for training:\n";
	cin >> *iterations;

	cout << "Number (int) of the k clusters (upto 50):\n";
	cin >> *k;
}

//-------------------User can define the number of: data points (N) and iterations for parallel implementation--------------------------------------
void input_user_parallel(std::string* infile_name, int* num, int* iterations){
	cout << "Number (int) of points you want to analyze (100, 1000, 10000, 100000, 1000000):\n";
	std::cin >> *num;
	int n = *num;
	
	switch (n){
	case 100: *infile_name = "option1_100_points.txt";
		break;
	case 1000: *infile_name = "option2_1000_points.txt";
		break;
	case 10000: *infile_name = "option3_10000_points.txt";
		break;
	case 100000: *infile_name = "option4_100000_points.txt";
		break;
	case 1000000: *infile_name = "option5_1000000_points.txt";
		break;
	default: *infile_name = "option1_100_points.txt";
		cout << "The Dataset with " << n << " points does not exist!\nThe \"option1_100_points.txt\" dataset is chosen instead of default.\n" << endl;
		break;
	}

	cout << "Please, insert number (int) of iterations for training:\n";
	cin >> *iterations;
}

//-----------------------------------------------------------------Driver Code----------------------------------------------------------------------
int main()
{
	std::string input_file;
	int N,MAX_ITERATION;

	int typeofexecution; //0-> sequential, 1-> parallel
	cout << "Please choose between sequential or parallel execution: 0 -> sequential, 1 -> parallel" << endl;
	cin >> typeofexecution;
	//------------------------------------------------Sequential Implementation---------------------------------------------------------------------
	if (typeofexecution == 0) {
		int K_seq=50,D_seq=2;
		input_user_seq(&input_file, &N, &K_seq, &MAX_ITERATION);
		// allocate memory 
		float datapoints[2 * 1000] = { 0 };
		int clust_assn[1000] = { 0 };
		float centroids[2 * 1000] = { 0 };
		int clust_sizes[55] = { 0 };

		srand(5);

		// initialize datapoints
		Read_from_file(datapoints, input_file);

		//initialize centroids
		centroid_initialization_seq(datapoints, centroids, N, K_seq);

		for (int c = 0; c < K_seq; ++c) {
			printf("Initialization of %d centroids: \n", K_seq);
			printf("(%f, %f)\n", centroids[2 * c], centroids[2 * c + 1]);
		}

		int current_iteration = 0;
		float time_assignments = 0;

		// Loop for the Iterations
		auto start_while = high_resolution_clock::now(); // for calculating the runtime
		while (current_iteration < MAX_ITERATION)
		{
			// cluster assignment for sequential implentation
			auto start = high_resolution_clock::now();
			kMeansClusterAssignment_seq(datapoints, clust_assn, centroids, N, K_seq);
			auto stop = high_resolution_clock::now();

			// getting the runtime of cluster assignment for sequential implentation
			auto duration = duration_cast<microseconds>(stop - start);
			float temp = duration.count();
			time_assignments = time_assignments + temp;

			// initialize cluster sizes (number of points in the cluster) back to zero
			for (int c = 0; c < K_seq; c++) {
				clust_sizes[c] = 0;
			}

			// initialize centroids back to zero
			for (int p = 0; p < D * K_seq; p++) {
				centroids[p] = 0.0;
			}

			// centroid update for sequential implentation
			kMeansCentroidUpdate_seq(datapoints, clust_assn, centroids, clust_sizes, N, K_seq);

			current_iteration += 1;
		}

		auto stop_while = high_resolution_clock::now(); // end of the runtime calculation for the total iterations

		// print the final centroids
		cout << "N = " << N << ",K = " << K_seq << ", Total ITERATION= " << MAX_ITERATION << ".\nThe centroids are:\n";
		for (int c = 0; c < K_seq; c++) {
			cout << "centroid: " << c << ": (" << centroids[2 * c] << ", " << centroids[2 * c + 1] << ")" << endl;
		}

		// get the runtime for the whole iteration runs
		auto duration_while = duration_cast<microseconds>(stop_while - start_while);
		float temp = duration_while.count();
		cout << "Time taken by " << MAX_ITERATION << " iterations is: " << temp << " microseconds" << endl;

		// the average runtime for assignments of the clusters for each data point over all the iterations
		time_assignments = time_assignments / MAX_ITERATION;
		//cout << "Average Time taken for Assigning the clusters for data points: " << time_assignments << " microseconds" << endl;

		// Naming of the output csv files for data points, centroids, and the clusters assignments
		std::string outfile_points = "D:\School\MasterDegree\CSS535\Final Project\KMeanClustering_Arsheya_Sunjil\KMeanClustering_Arsheya_Sunjil\outdir\datapoints.csv";
		std::string outfile_centroids = "D:\School\MasterDegree\CSS535\Final Project\KMeanClustering_Arsheya_Sunjil\KMeanClustering_Arsheya_Sunjil\outdir\centroids.csv";
		std::string outfile_clust = "D:\School\MasterDegree\CSS535\Final Project\KMeanClustering_Arsheya_Sunjil\KMeanClustering_Arsheya_Sunjil\outdir\clusters.csv";

		// Writing to the output files
		write2csv(datapoints, outfile_points, N);
		write2csv(centroids, outfile_centroids, K_seq);
		write2csv_clust(datapoints, clust_assn, outfile_clust, N);
	}
	//------------------------------------------------Parallel Implementation-----------------------------------------------------------------------
	else if (typeofexecution == 1){
		input_user_parallel(&input_file, &N, &MAX_ITERATION);
		//allocation of memory on the device 
		float* d_datapoints = 0;
		int* d_clust_assn = 0;
		float* d_centroids = 0;
		int* d_clust_sizes = 0;

		cudaMalloc(&d_datapoints, D * N * sizeof(float));
		cudaMalloc(&d_clust_assn, N * sizeof(int));
		cudaMalloc(&d_centroids, D * K * sizeof(float));
		cudaMalloc(&d_clust_sizes, K * sizeof(float));

		// allocation of memory in host
		float* h_centroids = (float*)malloc(D * K * sizeof(float));
		float* h_datapoints = (float*)malloc(D * N * sizeof(float));
		int* h_clust_sizes = (int*)malloc(K * sizeof(int));
		int* h_clust_assn = (int*)malloc(N * sizeof(int));

		srand(5);

		//initialize datapoints
		Read_from_file(h_datapoints, input_file);

		//initialize centroids
		centroid_initialization_parallel(h_datapoints, h_centroids, N);

		for (int c = 0; c < K; ++c) {
			printf("Initialization of %d centroids: \n", K);
			printf("(%f, %f)\n", h_centroids[2 * c], h_centroids[2 * c + 1]);
		}

		//initialize centroids counter for each cluster
		for (int c = 0; c < K; ++c) {
			h_clust_sizes[c] = 0;
		}

		// Calculation of runtime for transferring data from CPU to GPU
		auto start_cp0 = high_resolution_clock::now();
		cudaMemcpy(d_centroids, h_centroids, D * K * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(d_datapoints, h_datapoints, D * N * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(d_clust_sizes, h_clust_sizes, K * sizeof(int), cudaMemcpyHostToDevice);
		auto stop_cp0 = high_resolution_clock::now();

		// get and print the time for transfering data from CPU to GPU
		auto duration_cp0 = duration_cast<microseconds>(stop_cp0 - start_cp0);
		float temp = duration_cp0.count();
		//cout << "Time taken for transfering centroids, datapoints and cluster's sizes from host to device is : " << temp << " microseconds" << endl;

		int current_iteration = 0;

		float time_assignments = 0;					// total time for cluster assignment
		float time_copy = 0;						// total time for copying data from DtoH
		float time_copy_2 = 0;						// total time for copying data from HtoD

		// Calculation of the total run time for all the iterations
		auto start_while = high_resolution_clock::now();
		// Iteration for the k-means clustering algorithm
		while (current_iteration < MAX_ITERATION){
			// Runtime calculation for cluster assignment in parallel implementation
			auto start = high_resolution_clock::now();
			// Kernel call
			kMeansClusterAssignmentusingdistance_parallel <<<(N + TPB - 1) / TPB, TPB >> > (d_datapoints, d_clust_assn, d_centroids, N);
			auto stop = high_resolution_clock::now();

			// get the time for cluster assignment
			auto duration = duration_cast<microseconds>(stop - start);
			float temp = duration.count();
			time_assignments = time_assignments + temp;

			// Time taken for copying data (new centroids and cluster assignment) from GPU to CPU
			auto start_cp = high_resolution_clock::now();
			cudaMemcpy(h_centroids, d_centroids, D * K * sizeof(float), cudaMemcpyDeviceToHost);
			cudaMemcpy(h_clust_assn, d_clust_assn, N * sizeof(int), cudaMemcpyDeviceToHost);
			auto stop_cp = high_resolution_clock::now();

			// get the time for copying data from GPU to CPU
			auto duration_cp = duration_cast<microseconds>(stop_cp - start_cp);
			float temp_cp = duration_cp.count();
			time_copy = time_copy + temp_cp;

			//reset centroids and cluster sizes (will be updated in the next kernel)
			memset(h_centroids, 0.0, D * K * sizeof(float));
			memset(h_clust_sizes, 0, K * sizeof(int));

			//function call for centroid update
			//kMeansCentroidUpdate_parallel_new <<< (N + TPB - 1) / TPB, TPB >>> (h_datapoints, h_clust_assn, h_centroids, K, N, D);
			kMeansCentroidUpdate_parallel(h_datapoints, h_clust_assn, h_centroids, h_clust_sizes, N);

			// Time taken for copying centroid data from CPU to GPU for each iteration
			auto start_cp2 = high_resolution_clock::now();
			cudaMemcpy(d_centroids, h_centroids, D * K * sizeof(float), cudaMemcpyHostToDevice);
			auto stop_cp2 = high_resolution_clock::now();
			
			// Printing out the centroids for every iteration
			/*cout << "N = " << N << ",K = " << K << ", MAX_ITER= " << MAX_ITER << ".\nThe centroids are:\n";
			for (int l = 0; l < K; l++) {
				cout << "centroid: " << l << ": (" << h_centroids[2 * l] << ", " << h_centroids[2 * l + 1] << ")" << endl;
			}*/

			// getting the runtime for the updation of the centroids
			auto duration_cp2 = duration_cast<microseconds>(stop_cp2 - start_cp2);
			float temp_cp2 = duration_cp2.count();
			time_copy_2 = time_copy_2 + temp_cp2;

			current_iteration += 1;
		}

		auto stop_while = high_resolution_clock::now();	// end of the runtime calculation for the total iterations

		// Printing the final centroids
		cout << "N = " << N << ",K = " << K << ", Total ITERATION= " << MAX_ITERATION << ".\nThe centroids are:\n";
		for (int l = 0; l < K; l++) {
			cout << "centroid: " << l << ": (" << h_centroids[2 * l] << ", " << h_centroids[2 * l + 1] << ")" << endl;
		}

		// Printing the total time taken by all the iterations
		auto duration_while = duration_cast<microseconds>(stop_while - start_while);
		float temp_while = duration_while.count();
		cout << "Time taken by " << MAX_ITERATION << " iterations is: " << temp_while << " microseconds" << endl;

		// Printing the average time for cluster assignment during each iteration 
		time_assignments = time_assignments / MAX_ITERATION;
		//cout << "Time taken by kMeansClusterAssignment: " << time_assignments << " microseconds" << endl;

		// Printing the average time for copying DtoH during each iteration 
		time_copy = time_copy / MAX_ITERATION;
		//cout << "Time taken by transfering centroids and assignments from the device to the host: " << time_copy << " microseconds" << endl;

		// Printing the average time for copying HtoD during each iteration 
		time_copy_2 = time_copy_2 / MAX_ITERATION;
		//cout << "Time taken by transfering centroids and assignments from the device to the host: " << time_copy_2 << " microseconds" << endl;


		// Naming of the output csv files for data points, centroids, and the clusters assignments.
		std::string outfile_points = "D:\School\MasterDegree\CSS535\Final Project\KMeanClustering_Arsheya_Sunjil\KMeanClustering_Arsheya_Sunjil\outdir\datapoints.csv";
		std::string outfile_centroids = "D:\School\MasterDegree\CSS535\Final Project\KMeanClustering_Arsheya_Sunjil\KMeanClustering_Arsheya_Sunjil\outdir\centroids.csv";
		std::string outfile_clust = "D:\School\MasterDegree\CSS535\Final Project\KMeanClustering_Arsheya_Sunjil\KMeanClustering_Arsheya_Sunjil\outdir\clusters.csv";

		// Writing to the output files
		write2csv(h_datapoints, outfile_points, N);
		write2csv(h_centroids, outfile_centroids, K);
		write2csv_clust(h_datapoints, h_clust_assn, outfile_clust, N);

		// Cleanup memory on device
		cudaFree(d_datapoints);
		cudaFree(d_clust_assn);
		cudaFree(d_centroids);
		cudaFree(d_clust_sizes);

		// Cleanup memory on host
		free(h_centroids);
		free(h_datapoints);
		free(h_clust_sizes);
		free(h_clust_assn);
	}

	return 0;
}
