__device__ void mulq( const unsigned int *a,unsigned int *r){
    unsigned int u_2;
    unsigned int v_2;
    unsigned int carry,carry_word;

    for(int  i=0; i<NLIMBS32*2; i++)
        r[i]=0;

    for(int  i=0; i<NLIMBS32; i++) {
        carry=0;
        carry_word=0;

        #pragma unroll
        for(int j=0; j<NLIMBS32; j++) {
            __mull(v_2,a[i],q[j]);
            __mulh(u_2,a[i],q[j]);
            __add_cc(v_2,v_2,carry_word);
            __addc(carry_word,0,0);
            __add_cc(r[i+j],v_2,r[i+j]);
            __addc(carry_word,carry_word,carry);
            // __addc(carry,0,0);
            __add_cc(carry_word,u_2,carry_word);
            __addc(carry,0,0);
        }

        r[i+NLIMBS32]=carry_word;
    }
}

Checking "multiplication" with gmp ... incorrect calculation at iteration 0
        000000000000000000000000000000000000000000000000000000000000000100100000100000000001110110011111 = 4840234399
      * 000000000000000000000000000000000000000000000000000000000000000110011000001101100011100101110000 = 6848657776
 gpu: = 000000000000000000000000000000001010101110001001001011010101011101100110001011110101110010010000 = 12360460505694821520
 gmp: = 000000000000000000000000000000011100110000001001010010101111011001100110001011110101110010010000 = 33149108954374036624
