#include "timerc.h"
#include <math.h>

__host__ void merge(int *a, int left_start, int right_end) { // a is source array
    int left1, right1, left2, right2, k, i, j;
    int size = 1;
    int len_arr = right_end - left_start + 1;
    int* b =  (int *) malloc(sizeof(int) * len_arr);    // b is temp array

    while (size < len_arr) {
        k = left_start;
        left1 = left_start;
        while (left1 + size < len_arr) {
            right1 = left1 + size - 1;
            left2 = right1 + 1;
            right2 = left2 + size - 1;
            if (right2 >= len_arr) {
                right2 = len_arr - 1;
            }

            // start merging the list
            i = left1;
            j = left2;
            while (i <= right1 && j <= right2) {
                if (a[i] < a[j]) {
                    b[k] = a[i];
                    i++;
                    k++;
                    //printf("a[i] = %d\n", a[i]);
                }
                else {
                    b[k] = a[j];
                    j++;
                    k++;
                    //printf("a[i] = %d\n", a[i]);
                }
            }

            // if either left or right still has remaining
            while (i <= right1) {
                b[k] = a[i];
                i++;
                k++;
            }
            while (j <= right2) {
                b[k] = a[j];
                j++;
                k++;
            }

            //merge and sort other pairs
            left1 = right2 + 1;
        }
        // if there is any pair left that is unmerged
        i = left1;
        while (k < len_arr) {
            b[k] = a[i];
            i++;
            k++;
        }

        for (i = 0; i < len_arr; i++) {
            a[i] = b[i];
        }

        size = size * 2;
    }
    free(b);
}

__device__ __host__ void merge_gpu(int *a, int *b, int left_start, int right_end, int lev) { // a is source array, b is temp array
    int left1, right1, left2, right2, k, i, j;
    //int len_arr = right_end - left_start + 1;
    int size = pow(2, lev);

    k = left_start;
    left1 = left_start;
    right1 = left1 + size - 1;
    left2 = right1 + 1;
    right2 = left2 + size - 1;

    // start merging the list
    i = left1;
    j = left2;
    while (i <= right1 && j <= right2) {
        if (a[i] < a[j]) {
            b[k] = a[i];
            i++;
            k++;
        }
        else {
            b[k] = a[j];
            j++;
            k++;
        }
    }

    // if either left or right still has remaining
    while (i <= right1) {
        b[k] = a[i];
        i++;
        k++;
    }
    while (j <= right2) {
        b[k] = a[j];
        j++;
        k++;
    }
}

__global__ void gpu_mergesort_serial_merge(int *a, int *b, int level) {

    int thread_i  = threadIdx.x + blockIdx.x * blockDim.x;

    int left_start = pow(2,level) * 2 * thread_i;
    int right_end = pow(2,level) * ( 2 * thread_i + 1 ) + pow(2,level) - 1;

    merge_gpu(a, b, left_start, right_end, level);
}


void test_merge(){

    int a[8] ={4,2,4,7,9,1,3,8};
    int* b =  (int *) malloc(sizeof(int) * 8);
    merge_gpu(a, b, 0, 1, 0);
    merge_gpu(a, b, 2, 3, 0);
    merge_gpu(a, b, 4, 5, 0);
    merge_gpu(a, b, 6, 7, 0);

    printf("result should be {2,4,4,7,1,9,3,8}\n");
    printf("test result: \n");
    for (int i = 0; i < 8; i++){
        printf("%d ", b[i]);
    }
    printf("\n");
}


void test_pow_2(){
    printf("%lf\n", pow(2,3));
}

void test_log_2(){
    printf("%lf\n", log2(2));
}

int main() {
    /* test functions
    test_log_2();
    test_pow_2();
    test_merge();
    */

    int n = pow(2,13);
    int num_threads_per_block = 1;
    int* h_arr =  (int *) malloc(sizeof(int)*n);

    //generate an array with numbers
    for (int i = 0; i < n; i++) {
        h_arr[i] = (n-1) - i;
    }

    // call gpu_mergesort and generate gpu_result
    int * d_arr;
    int * d_temp;
    int * gpu_result = (int *) malloc( n * sizeof(int) );
    cudaMalloc( (void**) &d_arr, n * sizeof(int) );
    cudaMemcpy( d_arr, h_arr, n * sizeof(int), cudaMemcpyHostToDevice );
    cudaMalloc( (void**) &d_temp, n * sizeof(int) );

    float gpu_time;
    gstart();
    int flag;
    for (int lev = 0; lev < log2(n); lev++) {   // lev means level
        /*
        if ( ( n / num_threads_per_block / pow(2, lev+1) ) == 1) {
            num_threads_per_block = num_threads_per_block / 2;
        }
        */
        gpu_mergesort_serial_merge<<< n / num_threads_per_block / pow(2, lev+1) , num_threads_per_block >>>(d_arr, d_temp, lev);
        flag = 0;
        lev++;
        gpu_mergesort_serial_merge<<< n / num_threads_per_block / pow(2, lev+1) , num_threads_per_block >>>(d_temp, d_arr, lev);
        flag = 1;
    }
    gend(&gpu_time);
    printf("GPU time = %f\n",gpu_time);


    if (flag == 0) {    // finish on d_temp
        cudaMemcpy(gpu_result, d_temp, ( n * sizeof(int) ), cudaMemcpyDeviceToHost);
    }
    else {  // finish on d_arr
        cudaMemcpy(gpu_result, d_arr, ( n * sizeof(int) ), cudaMemcpyDeviceToHost);
    }

    // ----------------------------------------------------------------------------------------
    // merge sort on CPU
    float cpu_time;
    cstart();
    merge(h_arr, 0, n-1);
    cend(&cpu_time);
    printf("CPU time = %f\n",cpu_time);


    // debug
    /*
    printf("gpu result: \n");
    for (int i = 0; i < n; i++) {
        printf("%d ", gpu_result[i]);
    }
    printf("\n");

    // debug
    printf("cpu result: \n");
    for (int i = 0; i < n; i++) {
        printf("%d ", h_arr[i]);
    }
    printf("\n");
    */

    // compare cpu_result with gpu_result
    int error = 0;
    for (int i = 0; i < n; i++) {
        if (gpu_result[i] != h_arr[i]) {
            printf("ERROR\n");
            error = 1;
            break;
        }
    }

    if (error == 0) printf("gpu operation has the same result as the cpu operation\n");
    else printf("GPU and CPU results are different");

    cudaFree(d_temp);
    cudaFree(d_arr);
    free(h_arr);
}
