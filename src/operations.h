#include "bignum_types.h"


// Example of the schoolbook addition algorithm we will use if bignums were
// represented on 5 words:
//
//   A[4]---A[3]---A[2]---A[1]---A[0]
// + B[4]---B[3]---B[2]---B[1]---B[0]
// ------------------------------------
// | A[4] | A[3] | A[2] | A[1] | A[0] |
// |  +   |  +   |  +   |  +   |  +   |
// | B[4] | B[3] | B[2] | B[1] | B[0] |
// |  +   |  +   |  +   |  +   |      |
// | c_in | c_in | c_in | c_in |      |
// ------------------------------------
// | C[4] | C[3] | C[2] | C[1] | C[0] |
#define add_loc(c_loc, a_loc, b_loc)\
{\
    asm("add.cc.u32  %0, %1, %2;" : "=r"(c_loc[0]) : "r"(a_loc[0]), "r"(b_loc[0]));\
    asm("addc.u32    %0, %1, %2;" : "=r"(c_loc[1]) : "r"(a_loc[1]), "r"(b_loc[1]));\
}

#define add_glo(c_glo, a_glo, b_glo, tid)\
{\
    asm("add.cc.u32  %0, %1, %2;" : "=r"(c_glo[COAL_IDX(0, tid)]) : "r"(a_glo[COAL_IDX(0, tid)]), "r"(b_glo[COAL_IDX(0, tid)]));\
    asm("addc.u32    %0, %1, %2;" : "=r"(c_glo[COAL_IDX(1, tid)]) : "r"(a_glo[COAL_IDX(1, tid)]), "r"(b_glo[COAL_IDX(1, tid)]));\
}


// Example of the schoolbook subtraction algorithm we will use if bignums were
// represented on 5 words:
//
//   A[4]---A[3]---A[2]---A[1]---A[0]
// - B[4]---B[3]---B[2]---B[1]---B[0]
// ------------------------------------
// | A[4] | A[3] | A[2] | A[1] | A[0] |
// |  -   |  -   |  -   |  -   |  -   |
// | B[4] | B[3] | B[2] | B[1] | B[0] |
// |  -   |  -   |  -   |  -   |      |
// | b_in | b_in | b_in | b_in |      |
// ------------------------------------
// | C[4] | C[3] | C[2] | C[1] | C[0] |
#define sub_loc(c_loc, a_loc, b_loc)\
{\
    asm("sub.cc.u32  %0, %1, %2;" : "=r"(c_loc[0]) : "r"(a_loc[0]), "r"(b_loc[0]));\
    asm("subc.u32    %0, %1, %2;" : "=r"(c_loc[1]) : "r"(a_loc[1]), "r"(b_loc[1]));\
}

#define sub_glo(c_glo, a_glo, b_glo, tid)\
{\
    asm("sub.cc.u32  %0, %1, %2;" : "=r"(c_glo[COAL_IDX(0, tid)]) : "r"(a_glo[COAL_IDX(0, tid)]), "r"(b_glo[COAL_IDX(0, tid)]));\
    asm("subc.u32    %0, %1, %2;" : "=r"(c_glo[COAL_IDX(1, tid)]) : "r"(a_glo[COAL_IDX(1, tid)]), "r"(b_glo[COAL_IDX(1, tid)]));\
}


// Example of the schoolbook multiplication algorithm we will use if bignums
// were represented on 5 words:
//
//                                      A[4]---A[3]---A[2]---A[1]---A[0]
//                                    * B[4]---B[3]---B[2]---B[1]---B[0]
// -----------------------------------------------------------------------
// |      |      |      |      |      |      |      |      | B[0] * A[0] |
// |      |      |      |      |      |      |      | B[0] * A[1] |      |
// |      |      |      |      |      |      | B[0] * A[2] |      |      |
// |      |      |      |      |      | B[0] * A[3] |      |      |      |
// |      |      |      |      | B[0] * A[4] |      |      |      |      |
// |      |      |      |      |      |      |      | B[1] * A[0] |      |
// |      |      |      |      |      |      | B[1] * A[1] |      |      |
// |      |      |      |      |      | B[1] * A[2] |      |      |      |
// |      |      |      |      | B[1] * A[3] |      |      |      |      |
// |      |      |      | B[1] * A[4] |      |      |      |      |      |
// |      |      |      |      |      |      | B[2] * A[0] |      |      |
// |      |      |      |      |      | B[2] * A[1] |      |      |      |
// |      |      |      |      | B[2] * A[2] |      |      |      |      |
// |      |      |      | B[2] * A[3] |      |      |      |      |      |
// |      |      | B[2] * A[4] |      |      |      |      |      |      |
// |      |      |      |      |      | B[3] * A[0] |      |      |      |
// |      |      |      |      | B[3] * A[1] |      |      |      |      |
// |      |      |      | B[3] * A[2] |      |      |      |      |      |
// |      |      | B[3] * A[3] |      |      |      |      |      |      |
// |      | B[3] * A[4] |      |      |      |      |      |      |      |
// |      |      |      |      | B[4] * A[0] |      |      |      |      |
// |      |      |      | B[4] * A[1] |      |      |      |      |      |
// |      |      | B[4] * A[2] |      |      |      |      |      |      |
// |      | B[4] * A[3] |      |      |      |      |      |      |      |
// + B[4] * A[4] |      |      |      |      |      |      |      |      |
// -----------------------------------------------------------------------
// | C[9] | C[8] | C[7] | C[6] | C[5] | C[4] | C[3] | C[2] | C[1] | C[0] |
//
// Because of CUDA carry propagation problems (the carry flag is only kept for
// the next assembly instruction), we will have to order the steps in the
// following way:
//
//                                      A[4]---A[3]---A[2]---A[1]---A[0]
//                                    * B[4]---B[3]---B[2]---B[1]---B[0]
// -----------------------------------------------------------------------
// |      |      |      |      |      |      |      |      | B[0] * A[0] |
// |      |      |      |      |      |      |      | B[0] * A[1] |      |
// |      |      |      |      |      |      |      | B[1] * A[0] |      |
// |      |      |      |      |      |      | B[0] * A[2] |      |      |
// |      |      |      |      |      |      | B[1] * A[1] |      |      |
// |      |      |      |      |      |      | B[2] * A[0] |      |      |
// |      |      |      |      |      | B[0] * A[3] |      |      |      |
// |      |      |      |      |      | B[1] * A[2] |      |      |      |
// |      |      |      |      |      | B[2] * A[1] |      |      |      |
// |      |      |      |      |      | B[3] * A[0] |      |      |      |
// |      |      |      |      | B[0] * A[4] |      |      |      |      |
// |      |      |      |      | B[1] * A[3] |      |      |      |      |
// |      |      |      |      | B[2] * A[2] |      |      |      |      |
// |      |      |      |      | B[3] * A[1] |      |      |      |      |
// |      |      |      |      | B[4] * A[0] |      |      |      |      |
// |      |      |      | B[1] * A[4] |      |      |      |      |      |
// |      |      |      | B[2] * A[3] |      |      |      |      |      |
// |      |      |      | B[3] * A[2] |      |      |      |      |      |
// |      |      |      | B[4] * A[1] |      |      |      |      |      |
// |      |      | B[2] * A[4] |      |      |      |      |      |      |
// |      |      | B[3] * A[3] |      |      |      |      |      |      |
// |      |      | B[4] * A[2] |      |      |      |      |      |      |
// |      | B[3] * A[4] |      |      |      |      |      |      |      |
// |      | B[4] * A[3] |      |      |      |      |      |      |      |
// + B[4] * A[4] |      |      |      |      |      |      |      |      |
// -----------------------------------------------------------------------
// | C[9] | C[8] | C[7] | C[6] | C[5] | C[4] | C[3] | C[2] | C[1] | C[0] |
//
// Note: it is possible that C[9] will not be calculated if we are sure that the
// product of the 2 bignums will never require 2 * min_bignum_number_of_words
// words.
#define mul_loc(c_loc, a_loc, b_loc)\
{\
    {\
    uint32_t carry = 0;\
    asm("mul.lo.u32    %0, %1, %2    ;" : "=r"(c_loc[0]) : "r"(b_loc[0]), "r"(a_loc[0]));\
    asm("mul.hi.u32    %0, %1, %2    ;" : "=r"(c_loc[1]) : "r"(b_loc[0]), "r"(a_loc[0]));\
    asm("mad.lo.cc.u32 %0, %1, %2, %0;" : "+r"(c_loc[1]) : "r"(b_loc[0]), "r"(a_loc[1]));\
    asm("addc.u32      %0, %0,  0    ;" : "+r"(carry));\
    asm("mad.lo.cc.u32 %0, %1, %2, %0;" : "+r"(c_loc[1]) : "r"(b_loc[1]), "r"(a_loc[0]));\
    asm("addc.u32      %0, %0,  0    ;" : "+r"(carry));\
    asm("add.u32       %0, %1,  0    ;" : "=r"(c_loc[2]) : "r"(carry));\
    asm("add.u32       %0,  0,  0    ;" : "=r"(carry));\
    asm("mad.hi.cc.u32 %0, %1, %2, %0;" : "+r"(c_loc[2]) : "r"(b_loc[0]), "r"(a_loc[1]));\
    asm("addc.u32      %0, %0,  0    ;" : "+r"(carry));\
    asm("mad.hi.cc.u32 %0, %1, %2, %0;" : "+r"(c_loc[2]) : "r"(b_loc[1]), "r"(a_loc[0]));\
    asm("addc.u32      %0, %0,  0    ;" : "+r"(carry));\
    asm("mad.lo.cc.u32 %0, %1, %2, %0;" : "+r"(c_loc[2]) : "r"(b_loc[1]), "r"(a_loc[1]));\
    asm("addc.u32      %0, %0,  0    ;" : "+r"(carry));\
    asm("mad.hi.u32    %0, %1, %2, %3;" : "=r"(c_loc[3]) : "r"(b_loc[1]), "r"(a_loc[1]), "r"(carry));\
    }\
}

#define mul_karatsuba_loc(c_loc, a_loc, b_loc)\
{\
{\
uint32_t c0[2] = {0, 0};\
uint32_t c1[3] = {0, 0, 0};\
uint32_t c2[2] = {0, 0};\
uint32_t a0_plus_a1[2] = {0, 0};\
uint32_t b0_plus_b1[2] = {0, 0};\
{\
    {\
    uint32_t carry = 0;\
    asm("mul.lo.u32    %0, %1, %2    ;" : "=r"(c0[0]) : "r"(b_loc[0]), "r"(a_loc[0]));\
    asm("mad.hi.u32    %0, %1, %2, %3;" : "=r"(c0[1]) : "r"(b_loc[0]), "r"(a_loc[0]), "r"(carry));\
    }\
}\
{\
    {\
    uint32_t carry = 0;\
    asm("mul.lo.u32    %0, %1, %2    ;" : "=r"(c2[0]) : "r"(b_loc[1]), "r"(a_loc[1]));\
    asm("mad.hi.u32    %0, %1, %2, %3;" : "=r"(c2[1]) : "r"(b_loc[1]), "r"(a_loc[1]), "r"(carry));\
    }\
}\
    asm("add.cc.u32  %0, %1, %2;" : "=r"(a0_plus_a1[0]) : "r"(a_loc[0]), "r"(a_loc[1]));\
    asm("addc.u32    %0,  0,  0;" : "=r"(a0_plus_a1[1]) : );\
    asm("add.cc.u32  %0, %1, %2;" : "=r"(b0_plus_b1[0]) : "r"(b_loc[0]), "r"(b_loc[1]));\
    asm("addc.u32    %0,  0,  0;" : "=r"(b0_plus_b1[1]) : );\
{\
    {\
    uint32_t carry = 0;\
    asm("mul.lo.u32    %0, %1, %2    ;" : "=r"(c1[0]) : "r"(b0_plus_b1[0]), "r"(a0_plus_a1[0]));\
    asm("mul.hi.u32    %0, %1, %2    ;" : "=r"(c1[1]) : "r"(b0_plus_b1[0]), "r"(a0_plus_a1[0]));\
    asm("mad.lo.cc.u32 %0, %1, %2, %0;" : "+r"(c1[1]) : "r"(b0_plus_b1[0]), "r"(a0_plus_a1[1]));\
    asm("addc.u32      %0, %0,  0    ;" : "+r"(carry));\
    asm("mad.lo.cc.u32 %0, %1, %2, %0;" : "+r"(c1[1]) : "r"(b0_plus_b1[1]), "r"(a0_plus_a1[0]));\
    asm("addc.u32      %0, %0,  0    ;" : "+r"(carry));\
    asm("add.u32       %0, %1,  0    ;" : "=r"(c1[2]) : "r"(carry));\
    asm("add.u32       %0,  0,  0    ;" : "=r"(carry));\
    asm("mad.hi.cc.u32 %0, %1, %2, %0;" : "+r"(c1[2]) : "r"(b0_plus_b1[0]), "r"(a0_plus_a1[1]));\
    asm("addc.u32      %0, %0,  0    ;" : "+r"(carry));\
    asm("mad.hi.cc.u32 %0, %1, %2, %0;" : "+r"(c1[2]) : "r"(b0_plus_b1[1]), "r"(a0_plus_a1[0]));\
    asm("addc.u32      %0, %0,  0    ;" : "+r"(carry));\
    asm("mad.lo.cc.u32 %0, %1, %2, %0;" : "+r"(c1[2]) : "r"(b0_plus_b1[1]), "r"(a0_plus_a1[1]));\
    }\
}\
}\
}

#define mul_glo(c_glo, a_glo, b_glo, tid)\
{\
    {\
    uint32_t carry = 0;\
    asm("mul.lo.u32    %0, %1, %2    ;" : "=r"(c_glo[COAL_IDX(0, tid)]) : "r"(b_glo[COAL_IDX(0, tid)]), "r"(a_glo[COAL_IDX(0, tid)]));\
    asm("mul.hi.u32    %0, %1, %2    ;" : "=r"(c_glo[COAL_IDX(1, tid)]) : "r"(b_glo[COAL_IDX(0, tid)]), "r"(a_glo[COAL_IDX(0, tid)]));\
    asm("mad.lo.cc.u32 %0, %1, %2, %0;" : "+r"(c_glo[COAL_IDX(1, tid)]) : "r"(b_glo[COAL_IDX(0, tid)]), "r"(a_glo[COAL_IDX(1, tid)]));\
    asm("addc.u32      %0, %0,  0    ;" : "+r"(carry));\
    asm("mad.lo.cc.u32 %0, %1, %2, %0;" : "+r"(c_glo[COAL_IDX(1, tid)]) : "r"(b_glo[COAL_IDX(1, tid)]), "r"(a_glo[COAL_IDX(0, tid)]));\
    asm("addc.u32      %0, %0,  0    ;" : "+r"(carry));\
    asm("add.u32       %0, %1,  0    ;" : "=r"(c_glo[COAL_IDX(2, tid)]) : "r"(carry));\
    asm("add.u32       %0,  0,  0    ;" : "=r"(carry));\
    asm("mad.hi.cc.u32 %0, %1, %2, %0;" : "+r"(c_glo[COAL_IDX(2, tid)]) : "r"(b_glo[COAL_IDX(0, tid)]), "r"(a_glo[COAL_IDX(1, tid)]));\
    asm("addc.u32      %0, %0,  0    ;" : "+r"(carry));\
    asm("mad.hi.cc.u32 %0, %1, %2, %0;" : "+r"(c_glo[COAL_IDX(2, tid)]) : "r"(b_glo[COAL_IDX(1, tid)]), "r"(a_glo[COAL_IDX(0, tid)]));\
    asm("addc.u32      %0, %0,  0    ;" : "+r"(carry));\
    asm("mad.lo.cc.u32 %0, %1, %2, %0;" : "+r"(c_glo[COAL_IDX(2, tid)]) : "r"(b_glo[COAL_IDX(1, tid)]), "r"(a_glo[COAL_IDX(1, tid)]));\
    asm("addc.u32      %0, %0,  0    ;" : "+r"(carry));\
    asm("mad.hi.u32    %0, %1, %2, %3;" : "=r"(c_glo[COAL_IDX(3, tid)]) : "r"(b_glo[COAL_IDX(1, tid)]), "r"(a_glo[COAL_IDX(1, tid)]), "r"(carry));\
    }\
}

#define add_m_loc(c_loc, a_loc, b_loc, m_loc)\
{\
    uint32_t mask[2] = {0, 0};\
    asm("add.cc.u32  %0, %1, %2;" : "=r"(c_loc[0]) : "r"(a_loc[0]), "r"(b_loc[0]));\
    asm("addc.u32    %0, %1, %2;" : "=r"(c_loc[1]) : "r"(a_loc[1]), "r"(b_loc[1]));\
    asm("sub.cc.u32  %0, %1, %2;" : "=r"(c_loc[0]) : "r"(c_loc[0]), "r"(m_loc[0]));\
    asm("subc.cc.u32 %0, %1, %2;" : "=r"(c_loc[1]) : "r"(c_loc[1]), "r"(m_loc[1]));\
    asm("subc.cc.u32 %0, %1, %2;" : "=r"(mask[0]) : "r"(mask[0]), "r"(mask[0]));\
    asm("subc.u32    %0, %1, %2;" : "=r"(mask[1]) : "r"(mask[1]), "r"(mask[1]));\
    asm("and.b32     %0, %0, %1;" : "+r"(mask[0]) : "r"(m_loc[0]));\
    asm("and.b32     %0, %0, %1;" : "+r"(mask[1]) : "r"(m_loc[1]));\
    asm("add.cc.u32  %0, %1, %2;" : "=r"(c_loc[0]) : "r"(c_loc[0]), "r"(mask[0]));\
    asm("addc.u32    %0, %1, %2;" : "=r"(c_loc[1]) : "r"(c_loc[1]), "r"(mask[1]));\
}

#define sub_m_loc(c_loc, a_loc, b_loc, m_loc)\
{\
    uint32_t mask[2] = {0, 0};\
    asm("sub.cc.u32  %0, %1, %2;" : "=r"(c_loc[0]) : "r"(a_loc[0]), "r"(b_loc[0]));\
    asm("subc.cc.u32 %0, %1, %2;" : "=r"(c_loc[1]) : "r"(a_loc[1]), "r"(b_loc[1]));\
    asm("subc.cc.u32 %0, %1, %2;" : "=r"(mask[0]) : "r"(mask[0]), "r"(mask[0]));\
    asm("subc.u32    %0, %1, %2;" : "=r"(mask[1]) : "r"(mask[1]), "r"(mask[1]));\
    asm("and.b32     %0, %0, %1;" : "+r"(mask[0]) : "r"(m_loc[0]));\
    asm("and.b32     %0, %0, %1;" : "+r"(mask[1]) : "r"(m_loc[1]));\
    asm("add.cc.u32  %0, %1, %2;" : "=r"(c_loc[0]) : "r"(c_loc[0]), "r"(mask[0]));\
    asm("addc.u32    %0, %1, %2;" : "=r"(c_loc[1]) : "r"(c_loc[1]), "r"(mask[1]));\
}
