#if type=float || type=double   //Enable for types float and double only

{T} {T}_sig_helper({T} x) {
    return ({T})(1) / (({T})(1) + exp(-x));
}

kernel void {T}_sigmoid(global {T}* C, global {T}* B) {
    C[i] = {T}_sig_helper(B[i]);
}

kernel void {T}_sigmoid_in_place(global {T}* C) {
    C[i] = sig_helper(C[i]);
}

kernel void {T}_sigmoid_prime(global {T}* C, global {T}* B) {
    {T} sig = sig_helper(B[i]);
    C[i] = sig * (1.0 - sig);
}

kernel void {T}_sigmoid_prime_in_place(global {T}* C) {
    {T} sig = {T}_sig_helper(C[i]);
    C[i] = sig * (1.0 - sig);
}

#endif