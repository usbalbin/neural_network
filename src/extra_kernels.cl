{T} sig_helper({T} x) {
    return ({T})(1) / (({T})(1) + exp(-x));
}

kernel void sigmoid(global {T}* C, global {T}* B) {
    C[i] = sig_helper(B[i]);
}

kernel void sigmoid_in_place(global {T}* C) {
    C[i] = sig_helper(C[i]);
}

kernel void sigmoid_prime(global {T}* C, global {T}* B) {
    {T} sig = sig_helper(B[i]);
    C[i] = sig * (1.0 - sig);
}

kernel void sigmoid_prime_in_place(global {T}* C) {
    {T} sig = sig_helper(C[i]);
    C[i] = sig * (1.0 - sig);
}