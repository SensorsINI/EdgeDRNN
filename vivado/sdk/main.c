#include <stdio.h>
#include "platform.h"
#include "xil_printf.h"
#include "xil_types.h"
#include "xtime_l.h"
#include "edgedrnn.h"
#include "edgedrnn_test.h"
#include "dma.h"
#include "math.h"

#define VALIDATE_RNN 1
#define SUCCESS 0
#define FAIL 1

int serial_transfer(dma_t* p_dma_obj, short* p_snd_buf, short* p_rcv_buf);

//void overlap_transfer(dma_t* p_dma_obj, short* p_snd_buf, int buf_size);

int main()
{
	//----------------------------------------------
	// Create DMA Object
	//----------------------------------------------
	dma_t* p_dma_obj;
	p_dma_obj = dma_create(
		XPAR_AXI_DMA_0_DEVICE_ID,
		AQI,
		AQF,
		XPAR_AXI_DMA_0_M_AXI_MM2S_DATA_WIDTH/8,
		INP_SIZE,
		RNN_SIZE
	);

	int status = 0;
	short* p_snd_buf = NULL;
	short* p_rcv_buf = NULL;
	p_snd_buf = dma_set_snd_buf(p_dma_obj, p_snd_buf);
	p_rcv_buf = dma_set_rcv_buf(p_dma_obj, p_rcv_buf);

	//----------------------------------------------
	// Create Edgedrnn Object
	//----------------------------------------------
	edgedrnn_t* p_edgedrnn_obj = NULL;
	p_edgedrnn_obj = edgedrnn_create(
		XPAR_EDGEDRNN_WRAPPER_0_BASEADDR,
		(u32)rnn_param,
		NUM_PE,
		THX,		  // num_pe
		THH,		  // a_qi
		AQI,		  // a_qf
		AQF,		  // w_qi
		WQI,		  // w_qf
		WQF,
		XPAR_AXI_DMA_0_M_AXI_MM2S_DATA_WIDTH/8,	// hp_size
		RNN_LAYERS,                // rnn_num_layers
		INP_SIZE,	                // rnn_inp_size
		RNN_SIZE,	                // rnn_hid_size
		RNN_PARAM_SIZE                   // rnn_mat_size
	);

	//Set up the ARM PS
    init_platform();

    status = serial_transfer(p_dma_obj, p_snd_buf, p_rcv_buf);

#ifdef VALIDATE_RNN
    if (status == 0)
    {
    	printf("Benchmark Successful: RNN Outputs Correct!!!\r\n");
    } else {
    	printf("Benchmark Failed: RNN Outputs Wrong!!!\r\n");
    }
#endif

    free(p_snd_buf);
    free(p_rcv_buf);
    free(p_dma_obj);
    free(p_edgedrnn_obj);
	cleanup_platform();
    return 0;
}

int serial_transfer(dma_t* p_dma_obj, short* p_snd_buf, short* p_rcv_buf)
{
	int snd_pointer = 0; // Pointer to send data
	int rcv_pointer = 0; // Pointer to receive data
	int snd_buf_length = dma_get_snd_buf_length(p_dma_obj);
	int rcv_buf_length = dma_get_rcv_buf_length(p_dma_obj);
	int err_rnn = 0;


	//----------------------------------------------
	// Performance Evaluation
	//----------------------------------------------
	XTime tStart, tEnd;
	float step_ops = 2*(float)RNN_PARAM_SIZE;  // Number of operations in the network (per time step)
	float total_ops = step_ops*(float)EDGEDRNN_TEST_STIM_NUM_COLS; // Total ops in this test
	float total_latency = 0.;
	float min_latency = 0.;
	float max_latency = 0.;
	float mean_latency = 0.;
	float latency[EDGEDRNN_TEST_STIM_NUM_COLS] = {0.}; // Save latency of every
	int cl_out = 0;

	for(int i = 0; i < EDGEDRNN_TEST_STIM_NUM_COLS; i++)
	{
		XTime_GetTime(&tStart);
		dma_set_snd_buf_addr(p_dma_obj, (short*) (edgedrnn_test_stim + snd_pointer));
		dma_snd(p_dma_obj); // Kick-off MM2S Transfer
		dma_rcv(p_dma_obj); // Kick-off S2MM Transfer
		cl_out = classification_layer(p_rcv_buf);
		XTime_GetTime(&tEnd);


#ifdef VALIDATE_RNN
		// Validate RNN Outputs
		for (int j = 0; j < RNN_SIZE; j++)
		{
			err_rnn = *(p_rcv_buf + j) - edgedrnn_test_gold_rnn[j + rcv_pointer];
			if (err_rnn != 0)
				return FAIL;
		}

//		if (cl_out != edgedrnn_test_gold_fc[i])
//		{
//			printf("Error: Idx = %4d | cl_out = %d | gold = %d", i, cl_out, edgedrnn_test_gold_fc[i]);
//			return FAIL;
//		}
		rcv_pointer += rcv_buf_length; // Increment send pointer
#endif
		snd_pointer += snd_buf_length; // Increment send pointer
		latency[i] = 1.0 * (float)(tEnd - tStart) / ((float)(COUNTS_PER_SECOND)/(float)1000000);
	}

	// Get Total Latency
	min_latency = latency[0];
	max_latency = latency[0];
	for(int i = 0; i < EDGEDRNN_TEST_STIM_NUM_COLS; i++)
	{
		total_latency += latency[i];
		min_latency = min_latency > latency[i] ? latency[i] : min_latency;
		max_latency = max_latency < latency[i] ? latency[i] : max_latency;
	}
	mean_latency = total_latency/(float)EDGEDRNN_TEST_STIM_NUM_COLS;

	// Print Benchmark Results
	printf("Ops per Time Step            = %f\n\r", step_ops);
	printf("Total Time Steps             = %d\n\r", EDGEDRNN_TEST_STIM_NUM_COLS);
	printf("Total Ops                    = %f\n\r", total_ops);
	printf("Total Latency        (us)    = %f\n\r", total_latency);
	printf("Min Latency          (us)    = %f\n\r", min_latency);
	printf("Max Latency          (us)    = %f\n\r", max_latency);
	printf("Mean Latency         (us)    = %f\n\r", mean_latency);
	printf("Min Eff. Throughput  (GOp/s) = %f\n\r", step_ops/max_latency/1000.0);
	printf("Max Eff. Throughput  (GOp/s) = %f\n\r", step_ops/min_latency/1000.0);
	printf("Mean Eff. Throughput (GOp/s) = %f\n\r", step_ops/mean_latency/1000.0);

	return SUCCESS;
}


