#ifndef EDGEDRNN_PARAMS_H
#define EDGEDRNN_PARAMS_H
/*
 * Var Type: NN Parameter Matrix 
 * Var Name: rnn_param
 * Bit Width: 8
 * Dimension: (48, 58)
 */
#define RNN_PARAM_SIZE 2784
extern const signed char rnn_param[RNN_PARAM_SIZE] __attribute__ ((aligned (8)));
/*
 * Var Type: NN Parameter Matrix 
 * Var Name: cl_bias
 * Bit Width: 8
 * Dimension: (2, 1)
 */
#define CL_BIAS_SIZE 2
extern const signed char cl_bias[CL_BIAS_SIZE] __attribute__ ((aligned (8)));
/*
 * Var Type: NN Parameter Matrix 
 * Var Name: cl_weight
 * Bit Width: 8
 * Dimension: (2, 16)
 */
#define CL_WEIGHT_SIZE 32
extern const signed char cl_weight[CL_WEIGHT_SIZE] __attribute__ ((aligned (8)));
#define RNN_LAYERS 2
#define INP_SIZE 8
#define RNN_SIZE 16
#define THX 0
#define THH 0
#define NUM_PE 8
#define AQI 8
#define AQF 8
#define WQI 1
#define WQF 7
#endif