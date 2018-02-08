

#if defined(IS_FLOAT) || defined(IS_FLOAT)

//Sigmoid

{T} {T}_sig_helper({T} x) {
    return ({T})(1) / (({T})(1) + exp(-x));
}

kernel void {T}_sigmoid(global {T}* C, global {T}* B) {
    C[i] = {T}_sig_helper(B[i]);
}

kernel void {T}_sigmoid_in_place(global {T}* C) {
    C[i] = {T}_sig_helper(C[i]);
}

kernel void {T}_sigmoid_prime(global {T}* C, global {T}* B) {
    {T} sig = {T}_sig_helper(B[i]);
    C[i] = sig * (1.0 - sig);
}

kernel void {T}_sigmoid_prime_in_place(global {T}* C) {
    {T} sig = {T}_sig_helper(C[i]);
    C[i] = sig * (1.0 - sig);
}


//Relu
kernel void {T}_relu(global {T}* C, {T} A, global {T}* B) {
    C[i] = max(A, B[i]);
}

kernel void {T}_relu_in_place(global {T}* C, {T} A) {
    C[i] = max(A, C[i]);
}

kernel void {T}_relu_prime(global {T}* C, {T} A, global {T}* B) {
    C[i] = B[i] > 0 ? 1 : A;
}





#define lid get_local_id(0)
#define wgid get_group_id(0)
#define gz get_global_size(0)

/// Calculate the sum of all the elements in the vector
kernel void {T}_validate_sample(global const {T}* data, global const {T}* expected_data, global {T}* results, int count, local {T}* temp) {

	{T} value = 0.0;
	for (int globalIndex = i; globalIndex < count; globalIndex += gz) {
		value += fabs(data[globalIndex] - expected_data[globalIndex]) > 0.5 ? 0.0 : 1.0;
	}

	temp[lid] = value;
	barrier(CLK_LOCAL_MEM_FENCE);

	for (int offset = lz / 2; offset > 0; offset /= 2) {
		if (lid < offset)
			temp[lid] += temp[lid + offset];
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if (lid == 0) {
		results[wgid] = temp[0];
	}
}

#undef gz
#undef wgid
#undef lid

#endif

