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
    asm("addc.cc.u32 %0, %1, %2;" : "=r"(c_loc[1]) : "r"(a_loc[1]), "r"(b_loc[1]));\
    asm("addc.cc.u32 %0, %1, %2;" : "=r"(c_loc[2]) : "r"(a_loc[2]), "r"(b_loc[2]));\
    asm("addc.u32    %0, %1, %2;" : "=r"(c_loc[3]) : "r"(a_loc[3]), "r"(b_loc[3]));\
}

#define add_glo(c_glo, a_glo, b_glo, tid)\
{\
    asm("add.cc.u32  %0, %1, %2;" : "=r"(c_glo[COAL_IDX(0, tid)]) : "r"(a_glo[COAL_IDX(0, tid)]), "r"(b_glo[COAL_IDX(0, tid)]));\
    asm("addc.cc.u32 %0, %1, %2;" : "=r"(c_glo[COAL_IDX(1, tid)]) : "r"(a_glo[COAL_IDX(1, tid)]), "r"(b_glo[COAL_IDX(1, tid)]));\
    asm("addc.cc.u32 %0, %1, %2;" : "=r"(c_glo[COAL_IDX(2, tid)]) : "r"(a_glo[COAL_IDX(2, tid)]), "r"(b_glo[COAL_IDX(2, tid)]));\
    asm("addc.u32    %0, %1, %2;" : "=r"(c_glo[COAL_IDX(3, tid)]) : "r"(a_glo[COAL_IDX(3, tid)]), "r"(b_glo[COAL_IDX(3, tid)]));\
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
    asm("subc.cc.u32 %0, %1, %2;" : "=r"(c_loc[1]) : "r"(a_loc[1]), "r"(b_loc[1]));\
    asm("subc.cc.u32 %0, %1, %2;" : "=r"(c_loc[2]) : "r"(a_loc[2]), "r"(b_loc[2]));\
    asm("subc.u32    %0, %1, %2;" : "=r"(c_loc[3]) : "r"(a_loc[3]), "r"(b_loc[3]));\
}

#define sub_glo(c_glo, a_glo, b_glo, tid)\
{\
    asm("sub.cc.u32  %0, %1, %2;" : "=r"(c_glo[COAL_IDX(0, tid)]) : "r"(a_glo[COAL_IDX(0, tid)]), "r"(b_glo[COAL_IDX(0, tid)]));\
    asm("subc.cc.u32 %0, %1, %2;" : "=r"(c_glo[COAL_IDX(1, tid)]) : "r"(a_glo[COAL_IDX(1, tid)]), "r"(b_glo[COAL_IDX(1, tid)]));\
    asm("subc.cc.u32 %0, %1, %2;" : "=r"(c_glo[COAL_IDX(2, tid)]) : "r"(a_glo[COAL_IDX(2, tid)]), "r"(b_glo[COAL_IDX(2, tid)]));\
    asm("subc.u32    %0, %1, %2;" : "=r"(c_glo[COAL_IDX(3, tid)]) : "r"(a_glo[COAL_IDX(3, tid)]), "r"(b_glo[COAL_IDX(3, tid)]));\
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
    asm("mad.lo.cc.u32 %0, %1, %2, %0;" : "+r"(c_loc[2]) : "r"(b_loc[0]), "r"(a_loc[2]));\
    asm("addc.u32      %0, %0,  0    ;" : "+r"(carry));\
    asm("mad.lo.cc.u32 %0, %1, %2, %0;" : "+r"(c_loc[2]) : "r"(b_loc[1]), "r"(a_loc[1]));\
    asm("addc.u32      %0, %0,  0    ;" : "+r"(carry));\
    asm("mad.lo.cc.u32 %0, %1, %2, %0;" : "+r"(c_loc[2]) : "r"(b_loc[2]), "r"(a_loc[0]));\
    asm("addc.u32      %0, %0,  0    ;" : "+r"(carry));\
    asm("add.u32       %0, %1,  0    ;" : "=r"(c_loc[3]) : "r"(carry));\
    asm("add.u32       %0,  0,  0    ;" : "=r"(carry));\
    asm("mad.hi.cc.u32 %0, %1, %2, %0;" : "+r"(c_loc[3]) : "r"(b_loc[0]), "r"(a_loc[2]));\
    asm("addc.u32      %0, %0,  0    ;" : "+r"(carry));\
    asm("mad.hi.cc.u32 %0, %1, %2, %0;" : "+r"(c_loc[3]) : "r"(b_loc[1]), "r"(a_loc[1]));\
    asm("addc.u32      %0, %0,  0    ;" : "+r"(carry));\
    asm("mad.hi.cc.u32 %0, %1, %2, %0;" : "+r"(c_loc[3]) : "r"(b_loc[2]), "r"(a_loc[0]));\
    asm("addc.u32      %0, %0,  0    ;" : "+r"(carry));\
    asm("mad.lo.cc.u32 %0, %1, %2, %0;" : "+r"(c_loc[3]) : "r"(b_loc[0]), "r"(a_loc[3]));\
    asm("addc.u32      %0, %0,  0    ;" : "+r"(carry));\
    asm("mad.lo.cc.u32 %0, %1, %2, %0;" : "+r"(c_loc[3]) : "r"(b_loc[1]), "r"(a_loc[2]));\
    asm("addc.u32      %0, %0,  0    ;" : "+r"(carry));\
    asm("mad.lo.cc.u32 %0, %1, %2, %0;" : "+r"(c_loc[3]) : "r"(b_loc[2]), "r"(a_loc[1]));\
    asm("addc.u32      %0, %0,  0    ;" : "+r"(carry));\
    asm("mad.lo.cc.u32 %0, %1, %2, %0;" : "+r"(c_loc[3]) : "r"(b_loc[3]), "r"(a_loc[0]));\
    asm("addc.u32      %0, %0,  0    ;" : "+r"(carry));\
    asm("add.u32       %0, %1,  0    ;" : "=r"(c_loc[4]) : "r"(carry));\
    asm("add.u32       %0,  0,  0    ;" : "=r"(carry));\
    asm("mad.hi.cc.u32 %0, %1, %2, %0;" : "+r"(c_loc[4]) : "r"(b_loc[0]), "r"(a_loc[3]));\
    asm("addc.u32      %0, %0,  0    ;" : "+r"(carry));\
    asm("mad.hi.cc.u32 %0, %1, %2, %0;" : "+r"(c_loc[4]) : "r"(b_loc[1]), "r"(a_loc[2]));\
    asm("addc.u32      %0, %0,  0    ;" : "+r"(carry));\
    asm("mad.hi.cc.u32 %0, %1, %2, %0;" : "+r"(c_loc[4]) : "r"(b_loc[2]), "r"(a_loc[1]));\
    asm("addc.u32      %0, %0,  0    ;" : "+r"(carry));\
    asm("mad.hi.cc.u32 %0, %1, %2, %0;" : "+r"(c_loc[4]) : "r"(b_loc[3]), "r"(a_loc[0]));\
    asm("addc.u32      %0, %0,  0    ;" : "+r"(carry));\
    asm("mad.lo.cc.u32 %0, %1, %2, %0;" : "+r"(c_loc[4]) : "r"(b_loc[1]), "r"(a_loc[3]));\
    asm("addc.u32      %0, %0,  0    ;" : "+r"(carry));\
    asm("mad.lo.cc.u32 %0, %1, %2, %0;" : "+r"(c_loc[4]) : "r"(b_loc[2]), "r"(a_loc[2]));\
    asm("addc.u32      %0, %0,  0    ;" : "+r"(carry));\
    asm("mad.lo.cc.u32 %0, %1, %2, %0;" : "+r"(c_loc[4]) : "r"(b_loc[3]), "r"(a_loc[1]));\
    asm("addc.u32      %0, %0,  0    ;" : "+r"(carry));\
    asm("add.u32       %0, %1,  0    ;" : "=r"(c_loc[5]) : "r"(carry));\
    asm("add.u32       %0,  0,  0    ;" : "=r"(carry));\
    asm("mad.hi.cc.u32 %0, %1, %2, %0;" : "+r"(c_loc[5]) : "r"(b_loc[1]), "r"(a_loc[3]));\
    asm("addc.u32      %0, %0,  0    ;" : "+r"(carry));\
    asm("mad.hi.cc.u32 %0, %1, %2, %0;" : "+r"(c_loc[5]) : "r"(b_loc[2]), "r"(a_loc[2]));\
    asm("addc.u32      %0, %0,  0    ;" : "+r"(carry));\
    asm("mad.hi.cc.u32 %0, %1, %2, %0;" : "+r"(c_loc[5]) : "r"(b_loc[3]), "r"(a_loc[1]));\
    asm("addc.u32      %0, %0,  0    ;" : "+r"(carry));\
    asm("mad.lo.cc.u32 %0, %1, %2, %0;" : "+r"(c_loc[5]) : "r"(b_loc[2]), "r"(a_loc[3]));\
    asm("addc.u32      %0, %0,  0    ;" : "+r"(carry));\
    asm("mad.lo.cc.u32 %0, %1, %2, %0;" : "+r"(c_loc[5]) : "r"(b_loc[3]), "r"(a_loc[2]));\
    asm("addc.u32      %0, %0,  0    ;" : "+r"(carry));\
    asm("add.u32       %0, %1,  0    ;" : "=r"(c_loc[6]) : "r"(carry));\
    asm("add.u32       %0,  0,  0    ;" : "=r"(carry));\
    asm("mad.hi.cc.u32 %0, %1, %2, %0;" : "+r"(c_loc[6]) : "r"(b_loc[2]), "r"(a_loc[3]));\
    asm("addc.u32      %0, %0,  0    ;" : "+r"(carry));\
    asm("mad.hi.cc.u32 %0, %1, %2, %0;" : "+r"(c_loc[6]) : "r"(b_loc[3]), "r"(a_loc[2]));\
    asm("addc.u32      %0, %0,  0    ;" : "+r"(carry));\
    asm("mad.lo.cc.u32 %0, %1, %2, %0;" : "+r"(c_loc[6]) : "r"(b_loc[3]), "r"(a_loc[3]));\
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
    asm("mad.lo.cc.u32 %0, %1, %2, %0;" : "+r"(c_glo[COAL_IDX(2, tid)]) : "r"(b_glo[COAL_IDX(0, tid)]), "r"(a_glo[COAL_IDX(2, tid)]));\
    asm("addc.u32      %0, %0,  0    ;" : "+r"(carry));\
    asm("mad.lo.cc.u32 %0, %1, %2, %0;" : "+r"(c_glo[COAL_IDX(2, tid)]) : "r"(b_glo[COAL_IDX(1, tid)]), "r"(a_glo[COAL_IDX(1, tid)]));\
    asm("addc.u32      %0, %0,  0    ;" : "+r"(carry));\
    asm("mad.lo.cc.u32 %0, %1, %2, %0;" : "+r"(c_glo[COAL_IDX(2, tid)]) : "r"(b_glo[COAL_IDX(2, tid)]), "r"(a_glo[COAL_IDX(0, tid)]));\
    asm("addc.u32      %0, %0,  0    ;" : "+r"(carry));\
    asm("add.u32       %0, %1,  0    ;" : "=r"(c_glo[COAL_IDX(3, tid)]) : "r"(carry));\
    asm("add.u32       %0,  0,  0    ;" : "=r"(carry));\
    asm("mad.hi.cc.u32 %0, %1, %2, %0;" : "+r"(c_glo[COAL_IDX(3, tid)]) : "r"(b_glo[COAL_IDX(0, tid)]), "r"(a_glo[COAL_IDX(2, tid)]));\
    asm("addc.u32      %0, %0,  0    ;" : "+r"(carry));\
    asm("mad.hi.cc.u32 %0, %1, %2, %0;" : "+r"(c_glo[COAL_IDX(3, tid)]) : "r"(b_glo[COAL_IDX(1, tid)]), "r"(a_glo[COAL_IDX(1, tid)]));\
    asm("addc.u32      %0, %0,  0    ;" : "+r"(carry));\
    asm("mad.hi.cc.u32 %0, %1, %2, %0;" : "+r"(c_glo[COAL_IDX(3, tid)]) : "r"(b_glo[COAL_IDX(2, tid)]), "r"(a_glo[COAL_IDX(0, tid)]));\
    asm("addc.u32      %0, %0,  0    ;" : "+r"(carry));\
    asm("mad.lo.cc.u32 %0, %1, %2, %0;" : "+r"(c_glo[COAL_IDX(3, tid)]) : "r"(b_glo[COAL_IDX(0, tid)]), "r"(a_glo[COAL_IDX(3, tid)]));\
    asm("addc.u32      %0, %0,  0    ;" : "+r"(carry));\
    asm("mad.lo.cc.u32 %0, %1, %2, %0;" : "+r"(c_glo[COAL_IDX(3, tid)]) : "r"(b_glo[COAL_IDX(1, tid)]), "r"(a_glo[COAL_IDX(2, tid)]));\
    asm("addc.u32      %0, %0,  0    ;" : "+r"(carry));\
    asm("mad.lo.cc.u32 %0, %1, %2, %0;" : "+r"(c_glo[COAL_IDX(3, tid)]) : "r"(b_glo[COAL_IDX(2, tid)]), "r"(a_glo[COAL_IDX(1, tid)]));\
    asm("addc.u32      %0, %0,  0    ;" : "+r"(carry));\
    asm("mad.lo.cc.u32 %0, %1, %2, %0;" : "+r"(c_glo[COAL_IDX(3, tid)]) : "r"(b_glo[COAL_IDX(3, tid)]), "r"(a_glo[COAL_IDX(0, tid)]));\
    asm("addc.u32      %0, %0,  0    ;" : "+r"(carry));\
    asm("add.u32       %0, %1,  0    ;" : "=r"(c_glo[COAL_IDX(4, tid)]) : "r"(carry));\
    asm("add.u32       %0,  0,  0    ;" : "=r"(carry));\
    asm("mad.hi.cc.u32 %0, %1, %2, %0;" : "+r"(c_glo[COAL_IDX(4, tid)]) : "r"(b_glo[COAL_IDX(0, tid)]), "r"(a_glo[COAL_IDX(3, tid)]));\
    asm("addc.u32      %0, %0,  0    ;" : "+r"(carry));\
    asm("mad.hi.cc.u32 %0, %1, %2, %0;" : "+r"(c_glo[COAL_IDX(4, tid)]) : "r"(b_glo[COAL_IDX(1, tid)]), "r"(a_glo[COAL_IDX(2, tid)]));\
    asm("addc.u32      %0, %0,  0    ;" : "+r"(carry));\
    asm("mad.hi.cc.u32 %0, %1, %2, %0;" : "+r"(c_glo[COAL_IDX(4, tid)]) : "r"(b_glo[COAL_IDX(2, tid)]), "r"(a_glo[COAL_IDX(1, tid)]));\
    asm("addc.u32      %0, %0,  0    ;" : "+r"(carry));\
    asm("mad.hi.cc.u32 %0, %1, %2, %0;" : "+r"(c_glo[COAL_IDX(4, tid)]) : "r"(b_glo[COAL_IDX(3, tid)]), "r"(a_glo[COAL_IDX(0, tid)]));\
    asm("addc.u32      %0, %0,  0    ;" : "+r"(carry));\
    asm("mad.lo.cc.u32 %0, %1, %2, %0;" : "+r"(c_glo[COAL_IDX(4, tid)]) : "r"(b_glo[COAL_IDX(1, tid)]), "r"(a_glo[COAL_IDX(3, tid)]));\
    asm("addc.u32      %0, %0,  0    ;" : "+r"(carry));\
    asm("mad.lo.cc.u32 %0, %1, %2, %0;" : "+r"(c_glo[COAL_IDX(4, tid)]) : "r"(b_glo[COAL_IDX(2, tid)]), "r"(a_glo[COAL_IDX(2, tid)]));\
    asm("addc.u32      %0, %0,  0    ;" : "+r"(carry));\
    asm("mad.lo.cc.u32 %0, %1, %2, %0;" : "+r"(c_glo[COAL_IDX(4, tid)]) : "r"(b_glo[COAL_IDX(3, tid)]), "r"(a_glo[COAL_IDX(1, tid)]));\
    asm("addc.u32      %0, %0,  0    ;" : "+r"(carry));\
    asm("add.u32       %0, %1,  0    ;" : "=r"(c_glo[COAL_IDX(5, tid)]) : "r"(carry));\
    asm("add.u32       %0,  0,  0    ;" : "=r"(carry));\
    asm("mad.hi.cc.u32 %0, %1, %2, %0;" : "+r"(c_glo[COAL_IDX(5, tid)]) : "r"(b_glo[COAL_IDX(1, tid)]), "r"(a_glo[COAL_IDX(3, tid)]));\
    asm("addc.u32      %0, %0,  0    ;" : "+r"(carry));\
    asm("mad.hi.cc.u32 %0, %1, %2, %0;" : "+r"(c_glo[COAL_IDX(5, tid)]) : "r"(b_glo[COAL_IDX(2, tid)]), "r"(a_glo[COAL_IDX(2, tid)]));\
    asm("addc.u32      %0, %0,  0    ;" : "+r"(carry));\
    asm("mad.hi.cc.u32 %0, %1, %2, %0;" : "+r"(c_glo[COAL_IDX(5, tid)]) : "r"(b_glo[COAL_IDX(3, tid)]), "r"(a_glo[COAL_IDX(1, tid)]));\
    asm("addc.u32      %0, %0,  0    ;" : "+r"(carry));\
    asm("mad.lo.cc.u32 %0, %1, %2, %0;" : "+r"(c_glo[COAL_IDX(5, tid)]) : "r"(b_glo[COAL_IDX(2, tid)]), "r"(a_glo[COAL_IDX(3, tid)]));\
    asm("addc.u32      %0, %0,  0    ;" : "+r"(carry));\
    asm("mad.lo.cc.u32 %0, %1, %2, %0;" : "+r"(c_glo[COAL_IDX(5, tid)]) : "r"(b_glo[COAL_IDX(3, tid)]), "r"(a_glo[COAL_IDX(2, tid)]));\
    asm("addc.u32      %0, %0,  0    ;" : "+r"(carry));\
    asm("add.u32       %0, %1,  0    ;" : "=r"(c_glo[COAL_IDX(6, tid)]) : "r"(carry));\
    asm("add.u32       %0,  0,  0    ;" : "=r"(carry));\
    asm("mad.hi.cc.u32 %0, %1, %2, %0;" : "+r"(c_glo[COAL_IDX(6, tid)]) : "r"(b_glo[COAL_IDX(2, tid)]), "r"(a_glo[COAL_IDX(3, tid)]));\
    asm("addc.u32      %0, %0,  0    ;" : "+r"(carry));\
    asm("mad.hi.cc.u32 %0, %1, %2, %0;" : "+r"(c_glo[COAL_IDX(6, tid)]) : "r"(b_glo[COAL_IDX(3, tid)]), "r"(a_glo[COAL_IDX(2, tid)]));\
    asm("addc.u32      %0, %0,  0    ;" : "+r"(carry));\
    asm("mad.lo.cc.u32 %0, %1, %2, %0;" : "+r"(c_glo[COAL_IDX(6, tid)]) : "r"(b_glo[COAL_IDX(3, tid)]), "r"(a_glo[COAL_IDX(3, tid)]));\
    }\
}

#define mul_karatsuba_loc(c_loc, a_loc, b_loc)\
{\
    {\
    uint32_t c0[4] = {0, 0, 0, 0};\
    uint32_t c1[5] = {0, 0, 0, 0, 0};\
    uint32_t c2[3] = {0, 0, 0};\
    uint32_t a0[2] = {a_loc[0], a_loc[1]};\
    uint32_t b0[2] = {b_loc[0], b_loc[1]};\
    uint32_t a1[2] = {a_loc[2], a_loc[3]};\
    uint32_t b1[2] = {b_loc[2], b_loc[3]};\
    uint32_t a0_plus_a1[3] = {0, 0, 0};\
    uint32_t b0_plus_b1[3] = {0, 0, 0};\
    {\
    uint32_t carry = 0;\
    asm("mul.lo.u32    %0, %1, %2    ;" : "=r"(c0[0]) : "r"(b0[0]), "r"(a0[0]));\
    asm("mul.hi.u32    %0, %1, %2    ;" : "=r"(c0[1]) : "r"(b0[0]), "r"(a0[0]));\
    asm("mad.lo.cc.u32 %0, %1, %2, %0;" : "+r"(c0[1]) : "r"(b0[0]), "r"(a0[1]));\
    asm("addc.u32      %0, %0,  0    ;" : "+r"(carry));\
    asm("mad.lo.cc.u32 %0, %1, %2, %0;" : "+r"(c0[1]) : "r"(b0[1]), "r"(a0[0]));\
    asm("addc.u32      %0, %0,  0    ;" : "+r"(carry));\
    asm("add.u32       %0, %1,  0    ;" : "=r"(c0[2]) : "r"(carry));\
    asm("add.u32       %0,  0,  0    ;" : "=r"(carry));\
    asm("mad.hi.cc.u32 %0, %1, %2, %0;" : "+r"(c0[2]) : "r"(b0[0]), "r"(a0[1]));\
    asm("addc.u32      %0, %0,  0    ;" : "+r"(carry));\
    asm("mad.hi.cc.u32 %0, %1, %2, %0;" : "+r"(c0[2]) : "r"(b0[1]), "r"(a0[0]));\
    asm("addc.u32      %0, %0,  0    ;" : "+r"(carry));\
    asm("mad.lo.cc.u32 %0, %1, %2, %0;" : "+r"(c0[2]) : "r"(b0[1]), "r"(a0[1]));\
    asm("addc.u32      %0, %0,  0    ;" : "+r"(carry));\
    asm("mad.hi.u32    %0, %1, %2, %3;" : "=r"(c0[3]) : "r"(b0[1]), "r"(a0[1]), "r"(carry));\
    }\
    {\
    uint32_t carry = 0;\
    asm("mul.lo.u32    %0, %1, %2    ;" : "=r"(c2[0]) : "r"(b1[0]), "r"(a1[0]));\
    asm("mul.hi.u32    %0, %1, %2    ;" : "=r"(c2[1]) : "r"(b1[0]), "r"(a1[0]));\
    asm("mad.lo.cc.u32 %0, %1, %2, %0;" : "+r"(c2[1]) : "r"(b1[0]), "r"(a1[1]));\
    asm("addc.u32      %0, %0,  0    ;" : "+r"(carry));\
    asm("mad.lo.cc.u32 %0, %1, %2, %0;" : "+r"(c2[1]) : "r"(b1[1]), "r"(a1[0]));\
    asm("addc.u32      %0, %0,  0    ;" : "+r"(carry));\
    asm("add.u32       %0, %1,  0    ;" : "=r"(c2[2]) : "r"(carry));\
    asm("add.u32       %0,  0,  0    ;" : "=r"(carry));\
    asm("mad.hi.cc.u32 %0, %1, %2, %0;" : "+r"(c2[2]) : "r"(b1[0]), "r"(a1[1]));\
    asm("addc.u32      %0, %0,  0    ;" : "+r"(carry));\
    asm("mad.hi.cc.u32 %0, %1, %2, %0;" : "+r"(c2[2]) : "r"(b1[1]), "r"(a1[0]));\
    asm("addc.u32      %0, %0,  0    ;" : "+r"(carry));\
    asm("mad.lo.cc.u32 %0, %1, %2, %0;" : "+r"(c2[2]) : "r"(b1[1]), "r"(a1[1]));\
    }\
    asm("add.cc.u32  %0, %1, %2;" : "=r"(a0_plus_a1[0]) : "r"(a0[0]), "r"(a1[0]));\
    asm("addc.cc.u32 %0, %1, %2;" : "=r"(a0_plus_a1[1]) : "r"(a0[1]), "r"(a1[1]));\
    asm("addc.u32    %0,  0,  0;" : "=r"(a0_plus_a1[2]) : );\
    asm("add.cc.u32  %0, %1, %2;" : "=r"(b0_plus_b1[0]) : "r"(b0[0]), "r"(b1[0]));\
    asm("addc.cc.u32 %0, %1, %2;" : "=r"(b0_plus_b1[1]) : "r"(b0[1]), "r"(b1[1]));\
    asm("addc.u32    %0,  0,  0;" : "=r"(b0_plus_b1[2]) : );\
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
    asm("mad.lo.cc.u32 %0, %1, %2, %0;" : "+r"(c1[2]) : "r"(b0_plus_b1[0]), "r"(a0_plus_a1[2]));\
    asm("addc.u32      %0, %0,  0    ;" : "+r"(carry));\
    asm("mad.lo.cc.u32 %0, %1, %2, %0;" : "+r"(c1[2]) : "r"(b0_plus_b1[1]), "r"(a0_plus_a1[1]));\
    asm("addc.u32      %0, %0,  0    ;" : "+r"(carry));\
    asm("mad.lo.cc.u32 %0, %1, %2, %0;" : "+r"(c1[2]) : "r"(b0_plus_b1[2]), "r"(a0_plus_a1[0]));\
    asm("addc.u32      %0, %0,  0    ;" : "+r"(carry));\
    asm("add.u32       %0, %1,  0    ;" : "=r"(c1[3]) : "r"(carry));\
    asm("add.u32       %0,  0,  0    ;" : "=r"(carry));\
    asm("mad.hi.cc.u32 %0, %1, %2, %0;" : "+r"(c1[3]) : "r"(b0_plus_b1[0]), "r"(a0_plus_a1[2]));\
    asm("addc.u32      %0, %0,  0    ;" : "+r"(carry));\
    asm("mad.hi.cc.u32 %0, %1, %2, %0;" : "+r"(c1[3]) : "r"(b0_plus_b1[1]), "r"(a0_plus_a1[1]));\
    asm("addc.u32      %0, %0,  0    ;" : "+r"(carry));\
    asm("mad.hi.cc.u32 %0, %1, %2, %0;" : "+r"(c1[3]) : "r"(b0_plus_b1[2]), "r"(a0_plus_a1[0]));\
    asm("addc.u32      %0, %0,  0    ;" : "+r"(carry));\
    asm("mad.lo.cc.u32 %0, %1, %2, %0;" : "+r"(c1[3]) : "r"(b0_plus_b1[1]), "r"(a0_plus_a1[2]));\
    asm("addc.u32      %0, %0,  0    ;" : "+r"(carry));\
    asm("mad.lo.cc.u32 %0, %1, %2, %0;" : "+r"(c1[3]) : "r"(b0_plus_b1[2]), "r"(a0_plus_a1[1]));\
    asm("addc.u32      %0, %0,  0    ;" : "+r"(carry));\
    asm("add.u32       %0, %1,  0    ;" : "=r"(c1[4]) : "r"(carry));\
    asm("add.u32       %0,  0,  0    ;" : "=r"(carry));\
    asm("mad.hi.cc.u32 %0, %1, %2, %0;" : "+r"(c1[4]) : "r"(b0_plus_b1[1]), "r"(a0_plus_a1[2]));\
    asm("addc.u32      %0, %0,  0    ;" : "+r"(carry));\
    asm("mad.hi.cc.u32 %0, %1, %2, %0;" : "+r"(c1[4]) : "r"(b0_plus_b1[2]), "r"(a0_plus_a1[1]));\
    asm("addc.u32      %0, %0,  0    ;" : "+r"(carry));\
    asm("mad.lo.cc.u32 %0, %1, %2, %0;" : "+r"(c1[4]) : "r"(b0_plus_b1[2]), "r"(a0_plus_a1[2]));\
    }\
    asm("sub.cc.u32  %0, %1, %2;" : "=r"(c1[0]) : "r"(c1[0]), "r"(c0[0]));\
    asm("subc.cc.u32 %0, %1, %2;" : "=r"(c1[1]) : "r"(c1[1]), "r"(c0[1]));\
    asm("subc.cc.u32 %0, %1, %2;" : "=r"(c1[2]) : "r"(c1[2]), "r"(c0[2]));\
    asm("subc.cc.u32 %0, %1, %2;" : "=r"(c1[3]) : "r"(c1[3]), "r"(c0[3]));\
    asm("subc.u32    %0, %1,  0;" : "=r"(c1[4]) : "r"(c1[4]));\
    asm("sub.cc.u32  %0, %1, %2;" : "=r"(c1[0]) : "r"(c1[0]), "r"(c2[0]));\
    asm("subc.cc.u32 %0, %1, %2;" : "=r"(c1[1]) : "r"(c1[1]), "r"(c2[1]));\
    asm("subc.cc.u32 %0, %1, %2;" : "=r"(c1[2]) : "r"(c1[2]), "r"(c2[2]));\
    asm("subc.cc.u32 %0, %1,  0;" : "=r"(c1[3]) : "r"(c1[3]));\
    asm("subc.u32    %0, %1,  0;" : "=r"(c1[4]) : "r"(c1[4]));\
    asm("add.u32     %0, %1,  0;" : "=r"(c_loc[0]) : "r"(c0[0]));\
    asm("add.u32     %0, %1,  0;" : "=r"(c_loc[1]) : "r"(c0[1]));\
    asm("add.cc.u32  %0, %1, %2;" : "=r"(c_loc[2]) : "r"(c0[2]), "r"(c1[0]));\
    asm("addc.cc.u32 %0, %1, %2;" : "=r"(c_loc[3]) : "r"(c0[3]), "r"(c1[1]));\
    asm("addc.cc.u32 %0, %1, %2;" : "=r"(c_loc[4]) : "r"(c1[2]), "r"(c2[0]));\
    asm("addc.cc.u32 %0, %1, %2;" : "=r"(c_loc[5]) : "r"(c1[3]), "r"(c2[1]));\
    asm("addc.u32    %0, %1, %2;" : "=r"(c_loc[6]) : "r"(c1[4]), "r"(c2[2]));\
    }\
}


#define mul_karatsuba_glo(c_glo, a_glo, b_glo, tid)\
{\
    {\
    uint32_t c0[4] = {0, 0, 0, 0};\
    uint32_t c1[5] = {0, 0, 0, 0, 0};\
    uint32_t c2[3] = {0, 0, 0};\
    uint32_t a0[2] = {a_glo[COAL_IDX(0, tid)], a_glo[COAL_IDX(1, tid)]};\
    uint32_t b0[2] = {b_glo[COAL_IDX(0, tid)], b_glo[COAL_IDX(1, tid)]};\
    uint32_t a1[2] = {a_glo[COAL_IDX(2, tid)], a_glo[COAL_IDX(3, tid)]};\
    uint32_t b1[2] = {b_glo[COAL_IDX(2, tid)], b_glo[COAL_IDX(3, tid)]};\
    uint32_t a0_plus_a1[3] = {0, 0, 0};\
    uint32_t b0_plus_b1[3] = {0, 0, 0};\
    {\
    uint32_t carry = 0;\
    asm("mul.lo.u32    %0, %1, %2    ;" : "=r"(c0[0]) : "r"(b0[0]), "r"(a0[0]));\
    asm("mul.hi.u32    %0, %1, %2    ;" : "=r"(c0[1]) : "r"(b0[0]), "r"(a0[0]));\
    asm("mad.lo.cc.u32 %0, %1, %2, %0;" : "+r"(c0[1]) : "r"(b0[0]), "r"(a0[1]));\
    asm("addc.u32      %0, %0,  0    ;" : "+r"(carry));\
    asm("mad.lo.cc.u32 %0, %1, %2, %0;" : "+r"(c0[1]) : "r"(b0[1]), "r"(a0[0]));\
    asm("addc.u32      %0, %0,  0    ;" : "+r"(carry));\
    asm("add.u32       %0, %1,  0    ;" : "=r"(c0[2]) : "r"(carry));\
    asm("add.u32       %0,  0,  0    ;" : "=r"(carry));\
    asm("mad.hi.cc.u32 %0, %1, %2, %0;" : "+r"(c0[2]) : "r"(b0[0]), "r"(a0[1]));\
    asm("addc.u32      %0, %0,  0    ;" : "+r"(carry));\
    asm("mad.hi.cc.u32 %0, %1, %2, %0;" : "+r"(c0[2]) : "r"(b0[1]), "r"(a0[0]));\
    asm("addc.u32      %0, %0,  0    ;" : "+r"(carry));\
    asm("mad.lo.cc.u32 %0, %1, %2, %0;" : "+r"(c0[2]) : "r"(b0[1]), "r"(a0[1]));\
    asm("addc.u32      %0, %0,  0    ;" : "+r"(carry));\
    asm("mad.hi.u32    %0, %1, %2, %3;" : "=r"(c0[3]) : "r"(b0[1]), "r"(a0[1]), "r"(carry));\
    }\
    {\
    uint32_t carry = 0;\
    asm("mul.lo.u32    %0, %1, %2    ;" : "=r"(c2[0]) : "r"(b1[0]), "r"(a1[0]));\
    asm("mul.hi.u32    %0, %1, %2    ;" : "=r"(c2[1]) : "r"(b1[0]), "r"(a1[0]));\
    asm("mad.lo.cc.u32 %0, %1, %2, %0;" : "+r"(c2[1]) : "r"(b1[0]), "r"(a1[1]));\
    asm("addc.u32      %0, %0,  0    ;" : "+r"(carry));\
    asm("mad.lo.cc.u32 %0, %1, %2, %0;" : "+r"(c2[1]) : "r"(b1[1]), "r"(a1[0]));\
    asm("addc.u32      %0, %0,  0    ;" : "+r"(carry));\
    asm("add.u32       %0, %1,  0    ;" : "=r"(c2[2]) : "r"(carry));\
    asm("add.u32       %0,  0,  0    ;" : "=r"(carry));\
    asm("mad.hi.cc.u32 %0, %1, %2, %0;" : "+r"(c2[2]) : "r"(b1[0]), "r"(a1[1]));\
    asm("addc.u32      %0, %0,  0    ;" : "+r"(carry));\
    asm("mad.hi.cc.u32 %0, %1, %2, %0;" : "+r"(c2[2]) : "r"(b1[1]), "r"(a1[0]));\
    asm("addc.u32      %0, %0,  0    ;" : "+r"(carry));\
    asm("mad.lo.cc.u32 %0, %1, %2, %0;" : "+r"(c2[2]) : "r"(b1[1]), "r"(a1[1]));\
    }\
    asm("add.cc.u32  %0, %1, %2;" : "=r"(a0_plus_a1[0]) : "r"(a0[0]), "r"(a1[0]));\
    asm("addc.cc.u32 %0, %1, %2;" : "=r"(a0_plus_a1[1]) : "r"(a0[1]), "r"(a1[1]));\
    asm("addc.u32    %0,  0,  0;" : "=r"(a0_plus_a1[2]) : );\
    asm("add.cc.u32  %0, %1, %2;" : "=r"(b0_plus_b1[0]) : "r"(b0[0]), "r"(b1[0]));\
    asm("addc.cc.u32 %0, %1, %2;" : "=r"(b0_plus_b1[1]) : "r"(b0[1]), "r"(b1[1]));\
    asm("addc.u32    %0,  0,  0;" : "=r"(b0_plus_b1[2]) : );\
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
    asm("mad.lo.cc.u32 %0, %1, %2, %0;" : "+r"(c1[2]) : "r"(b0_plus_b1[0]), "r"(a0_plus_a1[2]));\
    asm("addc.u32      %0, %0,  0    ;" : "+r"(carry));\
    asm("mad.lo.cc.u32 %0, %1, %2, %0;" : "+r"(c1[2]) : "r"(b0_plus_b1[1]), "r"(a0_plus_a1[1]));\
    asm("addc.u32      %0, %0,  0    ;" : "+r"(carry));\
    asm("mad.lo.cc.u32 %0, %1, %2, %0;" : "+r"(c1[2]) : "r"(b0_plus_b1[2]), "r"(a0_plus_a1[0]));\
    asm("addc.u32      %0, %0,  0    ;" : "+r"(carry));\
    asm("add.u32       %0, %1,  0    ;" : "=r"(c1[3]) : "r"(carry));\
    asm("add.u32       %0,  0,  0    ;" : "=r"(carry));\
    asm("mad.hi.cc.u32 %0, %1, %2, %0;" : "+r"(c1[3]) : "r"(b0_plus_b1[0]), "r"(a0_plus_a1[2]));\
    asm("addc.u32      %0, %0,  0    ;" : "+r"(carry));\
    asm("mad.hi.cc.u32 %0, %1, %2, %0;" : "+r"(c1[3]) : "r"(b0_plus_b1[1]), "r"(a0_plus_a1[1]));\
    asm("addc.u32      %0, %0,  0    ;" : "+r"(carry));\
    asm("mad.hi.cc.u32 %0, %1, %2, %0;" : "+r"(c1[3]) : "r"(b0_plus_b1[2]), "r"(a0_plus_a1[0]));\
    asm("addc.u32      %0, %0,  0    ;" : "+r"(carry));\
    asm("mad.lo.cc.u32 %0, %1, %2, %0;" : "+r"(c1[3]) : "r"(b0_plus_b1[1]), "r"(a0_plus_a1[2]));\
    asm("addc.u32      %0, %0,  0    ;" : "+r"(carry));\
    asm("mad.lo.cc.u32 %0, %1, %2, %0;" : "+r"(c1[3]) : "r"(b0_plus_b1[2]), "r"(a0_plus_a1[1]));\
    asm("addc.u32      %0, %0,  0    ;" : "+r"(carry));\
    asm("add.u32       %0, %1,  0    ;" : "=r"(c1[4]) : "r"(carry));\
    asm("add.u32       %0,  0,  0    ;" : "=r"(carry));\
    asm("mad.hi.cc.u32 %0, %1, %2, %0;" : "+r"(c1[4]) : "r"(b0_plus_b1[1]), "r"(a0_plus_a1[2]));\
    asm("addc.u32      %0, %0,  0    ;" : "+r"(carry));\
    asm("mad.hi.cc.u32 %0, %1, %2, %0;" : "+r"(c1[4]) : "r"(b0_plus_b1[2]), "r"(a0_plus_a1[1]));\
    asm("addc.u32      %0, %0,  0    ;" : "+r"(carry));\
    asm("mad.lo.cc.u32 %0, %1, %2, %0;" : "+r"(c1[4]) : "r"(b0_plus_b1[2]), "r"(a0_plus_a1[2]));\
    }\
    asm("sub.cc.u32  %0, %1, %2;" : "=r"(c1[0]) : "r"(c1[0]), "r"(c0[0]));\
    asm("subc.cc.u32 %0, %1, %2;" : "=r"(c1[1]) : "r"(c1[1]), "r"(c0[1]));\
    asm("subc.cc.u32 %0, %1, %2;" : "=r"(c1[2]) : "r"(c1[2]), "r"(c0[2]));\
    asm("subc.cc.u32 %0, %1, %2;" : "=r"(c1[3]) : "r"(c1[3]), "r"(c0[3]));\
    asm("subc.u32    %0, %1,  0;" : "=r"(c1[4]) : "r"(c1[4]));\
    asm("sub.cc.u32  %0, %1, %2;" : "=r"(c1[0]) : "r"(c1[0]), "r"(c2[0]));\
    asm("subc.cc.u32 %0, %1, %2;" : "=r"(c1[1]) : "r"(c1[1]), "r"(c2[1]));\
    asm("subc.cc.u32 %0, %1, %2;" : "=r"(c1[2]) : "r"(c1[2]), "r"(c2[2]));\
    asm("subc.cc.u32 %0, %1,  0;" : "=r"(c1[3]) : "r"(c1[3]));\
    asm("subc.u32    %0, %1,  0;" : "=r"(c1[4]) : "r"(c1[4]));\
    asm("add.u32     %0, %1,  0;" : "=r"(c_glo[COAL_IDX(0, tid)]) : "r"(c0[0]));\
    asm("add.u32     %0, %1,  0;" : "=r"(c_glo[COAL_IDX(1, tid)]) : "r"(c0[1]));\
    asm("add.cc.u32  %0, %1, %2;" : "=r"(c_glo[COAL_IDX(2, tid)]) : "r"(c0[2]), "r"(c1[0]));\
    asm("addc.cc.u32 %0, %1, %2;" : "=r"(c_glo[COAL_IDX(3, tid)]) : "r"(c0[3]), "r"(c1[1]));\
    asm("addc.cc.u32 %0, %1, %2;" : "=r"(c_glo[COAL_IDX(4, tid)]) : "r"(c1[2]), "r"(c2[0]));\
    asm("addc.cc.u32 %0, %1, %2;" : "=r"(c_glo[COAL_IDX(5, tid)]) : "r"(c1[3]), "r"(c2[1]));\
    asm("addc.u32    %0, %1, %2;" : "=r"(c_glo[COAL_IDX(6, tid)]) : "r"(c1[4]), "r"(c2[2]));\
    }\
}


#define add_m_loc(c_loc, a_loc, b_loc, m_loc)\
{\
    uint32_t mask[4] = {0, 0, 0, 0};\
    asm("add.cc.u32  %0, %1, %2;" : "=r"(c_loc[0]) : "r"(a_loc[0]), "r"(b_loc[0]));\
    asm("addc.cc.u32 %0, %1, %2;" : "=r"(c_loc[1]) : "r"(a_loc[1]), "r"(b_loc[1]));\
    asm("addc.cc.u32 %0, %1, %2;" : "=r"(c_loc[2]) : "r"(a_loc[2]), "r"(b_loc[2]));\
    asm("addc.u32    %0, %1, %2;" : "=r"(c_loc[3]) : "r"(a_loc[3]), "r"(b_loc[3]));\
    asm("sub.cc.u32  %0, %1, %2;" : "=r"(c_loc[0]) : "r"(c_loc[0]), "r"(m_loc[0]));\
    asm("subc.cc.u32 %0, %1, %2;" : "=r"(c_loc[1]) : "r"(c_loc[1]), "r"(m_loc[1]));\
    asm("subc.cc.u32 %0, %1, %2;" : "=r"(c_loc[2]) : "r"(c_loc[2]), "r"(m_loc[2]));\
    asm("subc.cc.u32 %0, %1, %2;" : "=r"(c_loc[3]) : "r"(c_loc[3]), "r"(m_loc[3]));\
    asm("subc.cc.u32 %0, %1, %2;" : "=r"(mask[0]) : "r"(mask[0]), "r"(mask[0]));\
    asm("subc.cc.u32 %0, %1, %2;" : "=r"(mask[1]) : "r"(mask[1]), "r"(mask[1]));\
    asm("subc.cc.u32 %0, %1, %2;" : "=r"(mask[2]) : "r"(mask[2]), "r"(mask[2]));\
    asm("subc.u32    %0, %1, %2;" : "=r"(mask[3]) : "r"(mask[3]), "r"(mask[3]));\
    asm("and.b32     %0, %1, %2;" : "=r"(mask[0]) : "r"(mask[0]), "r"(m_loc[0]));\
    asm("and.b32     %0, %1, %2;" : "=r"(mask[1]) : "r"(mask[1]), "r"(m_loc[1]));\
    asm("and.b32     %0, %1, %2;" : "=r"(mask[2]) : "r"(mask[2]), "r"(m_loc[2]));\
    asm("and.b32     %0, %1, %2;" : "=r"(mask[3]) : "r"(mask[3]), "r"(m_loc[3]));\
    asm("add.cc.u32  %0, %1, %2;" : "=r"(c_loc[0]) : "r"(c_loc[0]), "r"(mask[0]));\
    asm("addc.cc.u32 %0, %1, %2;" : "=r"(c_loc[1]) : "r"(c_loc[1]), "r"(mask[1]));\
    asm("addc.cc.u32 %0, %1, %2;" : "=r"(c_loc[2]) : "r"(c_loc[2]), "r"(mask[2]));\
    asm("addc.u32    %0, %1, %2;" : "=r"(c_loc[3]) : "r"(c_loc[3]), "r"(mask[3]));\
}

#define add_m_glo(c_glo, a_glo, b_glo, m_glo, tid)\
{\
    uint32_t mask[4] = {0, 0, 0, 0};\
    asm("add.cc.u32  %0, %1, %2;" : "=r"(c_glo[COAL_IDX(0, tid)]) : "r"(a_glo[COAL_IDX(0, tid)]), "r"(b_glo[COAL_IDX(0, tid)]));\
    asm("addc.cc.u32 %0, %1, %2;" : "=r"(c_glo[COAL_IDX(1, tid)]) : "r"(a_glo[COAL_IDX(1, tid)]), "r"(b_glo[COAL_IDX(1, tid)]));\
    asm("addc.cc.u32 %0, %1, %2;" : "=r"(c_glo[COAL_IDX(2, tid)]) : "r"(a_glo[COAL_IDX(2, tid)]), "r"(b_glo[COAL_IDX(2, tid)]));\
    asm("addc.u32    %0, %1, %2;" : "=r"(c_glo[COAL_IDX(3, tid)]) : "r"(a_glo[COAL_IDX(3, tid)]), "r"(b_glo[COAL_IDX(3, tid)]));\
    asm("sub.cc.u32  %0, %1, %2;" : "=r"(c_glo[COAL_IDX(0, tid)]) : "r"(c_glo[COAL_IDX(0, tid)]), "r"(m_glo[COAL_IDX(0, tid)]));\
    asm("subc.cc.u32 %0, %1, %2;" : "=r"(c_glo[COAL_IDX(1, tid)]) : "r"(c_glo[COAL_IDX(1, tid)]), "r"(m_glo[COAL_IDX(1, tid)]));\
    asm("subc.cc.u32 %0, %1, %2;" : "=r"(c_glo[COAL_IDX(2, tid)]) : "r"(c_glo[COAL_IDX(2, tid)]), "r"(m_glo[COAL_IDX(2, tid)]));\
    asm("subc.cc.u32 %0, %1, %2;" : "=r"(c_glo[COAL_IDX(3, tid)]) : "r"(c_glo[COAL_IDX(3, tid)]), "r"(m_glo[COAL_IDX(3, tid)]));\
    asm("subc.cc.u32 %0, %1, %2;" : "=r"(mask[0]) : "r"(mask[0]), "r"(mask[0]));\
    asm("subc.cc.u32 %0, %1, %2;" : "=r"(mask[1]) : "r"(mask[1]), "r"(mask[1]));\
    asm("subc.cc.u32 %0, %1, %2;" : "=r"(mask[2]) : "r"(mask[2]), "r"(mask[2]));\
    asm("subc.u32    %0, %1, %2;" : "=r"(mask[3]) : "r"(mask[3]), "r"(mask[3]));\
    asm("and.b32     %0, %1, %2;" : "=r"(mask[0]) : "r"(mask[0]), "r"(m_glo[COAL_IDX(0, tid)]));\
    asm("and.b32     %0, %1, %2;" : "=r"(mask[1]) : "r"(mask[1]), "r"(m_glo[COAL_IDX(1, tid)]));\
    asm("and.b32     %0, %1, %2;" : "=r"(mask[2]) : "r"(mask[2]), "r"(m_glo[COAL_IDX(2, tid)]));\
    asm("and.b32     %0, %1, %2;" : "=r"(mask[3]) : "r"(mask[3]), "r"(m_glo[COAL_IDX(3, tid)]));\
    asm("add.cc.u32  %0, %1, %2;" : "=r"(c_glo[COAL_IDX(0, tid)]) : "r"(c_glo[COAL_IDX(0, tid)]), "r"(mask[0]));\
    asm("addc.cc.u32 %0, %1, %2;" : "=r"(c_glo[COAL_IDX(1, tid)]) : "r"(c_glo[COAL_IDX(1, tid)]), "r"(mask[1]));\
    asm("addc.cc.u32 %0, %1, %2;" : "=r"(c_glo[COAL_IDX(2, tid)]) : "r"(c_glo[COAL_IDX(2, tid)]), "r"(mask[2]));\
    asm("addc.u32    %0, %1, %2;" : "=r"(c_glo[COAL_IDX(3, tid)]) : "r"(c_glo[COAL_IDX(3, tid)]), "r"(mask[3]));\
}

#define sub_m_loc(c_loc, a_loc, b_loc, m_loc)\
{\
    uint32_t mask[4] = {0, 0, 0, 0};\
    asm("sub.cc.u32  %0, %1, %2;" : "=r"(c_loc[0]) : "r"(a_loc[0]), "r"(b_loc[0]));\
    asm("subc.cc.u32 %0, %1, %2;" : "=r"(c_loc[1]) : "r"(a_loc[1]), "r"(b_loc[1]));\
    asm("subc.cc.u32 %0, %1, %2;" : "=r"(c_loc[2]) : "r"(a_loc[2]), "r"(b_loc[2]));\
    asm("subc.cc.u32 %0, %1, %2;" : "=r"(c_loc[3]) : "r"(a_loc[3]), "r"(b_loc[3]));\
    asm("subc.cc.u32 %0, %1, %2;" : "=r"(mask[0]) : "r"(mask[0]), "r"(mask[0]));\
    asm("subc.cc.u32 %0, %1, %2;" : "=r"(mask[1]) : "r"(mask[1]), "r"(mask[1]));\
    asm("subc.cc.u32 %0, %1, %2;" : "=r"(mask[2]) : "r"(mask[2]), "r"(mask[2]));\
    asm("subc.u32    %0, %1, %2;" : "=r"(mask[3]) : "r"(mask[3]), "r"(mask[3]));\
    asm("and.b32     %0, %1, %2;" : "=r"(mask[0]) : "r"(mask[0]), "r"(m_loc[0]));\
    asm("and.b32     %0, %1, %2;" : "=r"(mask[1]) : "r"(mask[1]), "r"(m_loc[1]));\
    asm("and.b32     %0, %1, %2;" : "=r"(mask[2]) : "r"(mask[2]), "r"(m_loc[2]));\
    asm("and.b32     %0, %1, %2;" : "=r"(mask[3]) : "r"(mask[3]), "r"(m_loc[3]));\
    asm("add.cc.u32  %0, %1, %2;" : "=r"(c_loc[0]) : "r"(c_loc[0]), "r"(mask[0]));\
    asm("addc.cc.u32 %0, %1, %2;" : "=r"(c_loc[1]) : "r"(c_loc[1]), "r"(mask[1]));\
    asm("addc.cc.u32 %0, %1, %2;" : "=r"(c_loc[2]) : "r"(c_loc[2]), "r"(mask[2]));\
    asm("addc.u32    %0, %1, %2;" : "=r"(c_loc[3]) : "r"(c_loc[3]), "r"(mask[3]));\
}

#define sub_m_glo(c_glo, a_glo, b_glo, m_glo, tid)\
{\
    uint32_t mask[4] = {0, 0, 0, 0};\
    asm("sub.cc.u32  %0, %1, %2;" : "=r"(c_glo[COAL_IDX(0, tid)]) : "r"(a_glo[COAL_IDX(0, tid)]), "r"(b_glo[COAL_IDX(0, tid)]));\
    asm("subc.cc.u32 %0, %1, %2;" : "=r"(c_glo[COAL_IDX(1, tid)]) : "r"(a_glo[COAL_IDX(1, tid)]), "r"(b_glo[COAL_IDX(1, tid)]));\
    asm("subc.cc.u32 %0, %1, %2;" : "=r"(c_glo[COAL_IDX(2, tid)]) : "r"(a_glo[COAL_IDX(2, tid)]), "r"(b_glo[COAL_IDX(2, tid)]));\
    asm("subc.cc.u32 %0, %1, %2;" : "=r"(c_glo[COAL_IDX(3, tid)]) : "r"(a_glo[COAL_IDX(3, tid)]), "r"(b_glo[COAL_IDX(3, tid)]));\
    asm("subc.cc.u32 %0, %1, %2;" : "=r"(mask[0]) : "r"(mask[0]), "r"(mask[0]));\
    asm("subc.cc.u32 %0, %1, %2;" : "=r"(mask[1]) : "r"(mask[1]), "r"(mask[1]));\
    asm("subc.cc.u32 %0, %1, %2;" : "=r"(mask[2]) : "r"(mask[2]), "r"(mask[2]));\
    asm("subc.u32    %0, %1, %2;" : "=r"(mask[3]) : "r"(mask[3]), "r"(mask[3]));\
    asm("and.b32     %0, %1, %2;" : "=r"(mask[0]) : "r"(mask[0]), "r"(m_glo[COAL_IDX(0, tid)]));\
    asm("and.b32     %0, %1, %2;" : "=r"(mask[1]) : "r"(mask[1]), "r"(m_glo[COAL_IDX(1, tid)]));\
    asm("and.b32     %0, %1, %2;" : "=r"(mask[2]) : "r"(mask[2]), "r"(m_glo[COAL_IDX(2, tid)]));\
    asm("and.b32     %0, %1, %2;" : "=r"(mask[3]) : "r"(mask[3]), "r"(m_glo[COAL_IDX(3, tid)]));\
    asm("add.cc.u32  %0, %1, %2;" : "=r"(c_glo[COAL_IDX(0, tid)]) : "r"(c_glo[COAL_IDX(0, tid)]), "r"(mask[0]));\
    asm("addc.cc.u32 %0, %1, %2;" : "=r"(c_glo[COAL_IDX(1, tid)]) : "r"(c_glo[COAL_IDX(1, tid)]), "r"(mask[1]));\
    asm("addc.cc.u32 %0, %1, %2;" : "=r"(c_glo[COAL_IDX(2, tid)]) : "r"(c_glo[COAL_IDX(2, tid)]), "r"(mask[2]));\
    asm("addc.u32    %0, %1, %2;" : "=r"(c_glo[COAL_IDX(3, tid)]) : "r"(c_glo[COAL_IDX(3, tid)]), "r"(mask[3]));\
}

#define montgomery_reduction(c_loc, T_loc, m_loc, m_prime)\
{\
    uint32_t A[8] = {T_loc[0], T_loc[1], T_loc[2], T_loc[3], T_loc[4], T_loc[5], T_loc[6], 0};\
    uint32_t u_i_m[5] = {0, 0, 0, 0, 0};\
    uint32_t u_i[1] = {0};\
    asm("mul.lo.u32 %0, %1, %2;" : "=r"(u_i[0]) : "r"(A[0]), "r"(m_prime));\
    {\
    uint32_t carry = 0;\
    asm("mul.lo.u32    %0, %1, %2    ;" : "=r"(u_i_m[0]) : "r"(m_loc[0]), "r"(u_i[0]));\
    asm("mul.hi.u32    %0, %1, %2    ;" : "=r"(u_i_m[1]) : "r"(m_loc[0]), "r"(u_i[0]));\
    asm("mad.lo.cc.u32 %0, %1, %2, %0;" : "+r"(u_i_m[1]) : "r"(m_loc[1]), "r"(u_i[0]));\
    asm("addc.u32      %0, %0,  0    ;" : "+r"(carry));\
    asm("add.u32       %0, %1,  0    ;" : "=r"(u_i_m[2]) : "r"(carry));\
    asm("add.u32       %0,  0,  0    ;" : "=r"(carry));\
    asm("mad.hi.cc.u32 %0, %1, %2, %0;" : "+r"(u_i_m[2]) : "r"(m_loc[1]), "r"(u_i[0]));\
    asm("addc.u32      %0, %0,  0    ;" : "+r"(carry));\
    asm("mad.lo.cc.u32 %0, %1, %2, %0;" : "+r"(u_i_m[2]) : "r"(m_loc[2]), "r"(u_i[0]));\
    asm("addc.u32      %0, %0,  0    ;" : "+r"(carry));\
    asm("add.u32       %0, %1,  0    ;" : "=r"(u_i_m[3]) : "r"(carry));\
    asm("add.u32       %0,  0,  0    ;" : "=r"(carry));\
    asm("mad.hi.cc.u32 %0, %1, %2, %0;" : "+r"(u_i_m[3]) : "r"(m_loc[2]), "r"(u_i[0]));\
    asm("addc.u32      %0, %0,  0    ;" : "+r"(carry));\
    asm("mad.lo.cc.u32 %0, %1, %2, %0;" : "+r"(u_i_m[3]) : "r"(m_loc[3]), "r"(u_i[0]));\
    asm("addc.u32      %0, %0,  0    ;" : "+r"(carry));\
    asm("mad.hi.u32    %0, %1, %2, %3;" : "=r"(u_i_m[4]) : "r"(m_loc[3]), "r"(u_i[0]), "r"(carry));\
    }\
    asm("add.cc.u32  %0, %1, %2;" : "=r"(A[0]) : "r"(A[0]), "r"(u_i_m[0]));\
    asm("addc.cc.u32 %0, %1, %2;" : "=r"(A[1]) : "r"(A[1]), "r"(u_i_m[1]));\
    asm("addc.cc.u32 %0, %1, %2;" : "=r"(A[2]) : "r"(A[2]), "r"(u_i_m[2]));\
    asm("addc.cc.u32 %0, %1, %2;" : "=r"(A[3]) : "r"(A[3]), "r"(u_i_m[3]));\
    asm("addc.cc.u32 %0, %1, %2;" : "=r"(A[4]) : "r"(A[4]), "r"(u_i_m[4]));\
    asm("addc.cc.u32 %0, %1,  0;" : "=r"(A[5]) : "r"(A[5]));\
    asm("addc.cc.u32 %0, %1,  0;" : "=r"(A[6]) : "r"(A[6]));\
    asm("addc.u32    %0, %1,  0;" : "=r"(A[7]) : "r"(A[7]));\
    asm("mul.lo.u32 %0, %1, %2;" : "=r"(u_i[0]) : "r"(A[1]), "r"(m_prime));\
    {\
    uint32_t carry = 0;\
    asm("mul.lo.u32    %0, %1, %2    ;" : "=r"(u_i_m[0]) : "r"(m_loc[0]), "r"(u_i[0]));\
    asm("mul.hi.u32    %0, %1, %2    ;" : "=r"(u_i_m[1]) : "r"(m_loc[0]), "r"(u_i[0]));\
    asm("mad.lo.cc.u32 %0, %1, %2, %0;" : "+r"(u_i_m[1]) : "r"(m_loc[1]), "r"(u_i[0]));\
    asm("addc.u32      %0, %0,  0    ;" : "+r"(carry));\
    asm("add.u32       %0, %1,  0    ;" : "=r"(u_i_m[2]) : "r"(carry));\
    asm("add.u32       %0,  0,  0    ;" : "=r"(carry));\
    asm("mad.hi.cc.u32 %0, %1, %2, %0;" : "+r"(u_i_m[2]) : "r"(m_loc[1]), "r"(u_i[0]));\
    asm("addc.u32      %0, %0,  0    ;" : "+r"(carry));\
    asm("mad.lo.cc.u32 %0, %1, %2, %0;" : "+r"(u_i_m[2]) : "r"(m_loc[2]), "r"(u_i[0]));\
    asm("addc.u32      %0, %0,  0    ;" : "+r"(carry));\
    asm("add.u32       %0, %1,  0    ;" : "=r"(u_i_m[3]) : "r"(carry));\
    asm("add.u32       %0,  0,  0    ;" : "=r"(carry));\
    asm("mad.hi.cc.u32 %0, %1, %2, %0;" : "+r"(u_i_m[3]) : "r"(m_loc[2]), "r"(u_i[0]));\
    asm("addc.u32      %0, %0,  0    ;" : "+r"(carry));\
    asm("mad.lo.cc.u32 %0, %1, %2, %0;" : "+r"(u_i_m[3]) : "r"(m_loc[3]), "r"(u_i[0]));\
    asm("addc.u32      %0, %0,  0    ;" : "+r"(carry));\
    asm("mad.hi.u32    %0, %1, %2, %3;" : "=r"(u_i_m[4]) : "r"(m_loc[3]), "r"(u_i[0]), "r"(carry));\
    }\
    asm("add.cc.u32  %0, %1, %2;" : "=r"(A[1]) : "r"(A[1]), "r"(u_i_m[0]));\
    asm("addc.cc.u32 %0, %1, %2;" : "=r"(A[2]) : "r"(A[2]), "r"(u_i_m[1]));\
    asm("addc.cc.u32 %0, %1, %2;" : "=r"(A[3]) : "r"(A[3]), "r"(u_i_m[2]));\
    asm("addc.cc.u32 %0, %1, %2;" : "=r"(A[4]) : "r"(A[4]), "r"(u_i_m[3]));\
    asm("addc.cc.u32 %0, %1, %2;" : "=r"(A[5]) : "r"(A[5]), "r"(u_i_m[4]));\
    asm("addc.cc.u32 %0, %1,  0;" : "=r"(A[6]) : "r"(A[6]));\
    asm("addc.u32    %0, %1,  0;" : "=r"(A[7]) : "r"(A[7]));\
    asm("mul.lo.u32 %0, %1, %2;" : "=r"(u_i[0]) : "r"(A[2]), "r"(m_prime));\
    {\
    uint32_t carry = 0;\
    asm("mul.lo.u32    %0, %1, %2    ;" : "=r"(u_i_m[0]) : "r"(m_loc[0]), "r"(u_i[0]));\
    asm("mul.hi.u32    %0, %1, %2    ;" : "=r"(u_i_m[1]) : "r"(m_loc[0]), "r"(u_i[0]));\
    asm("mad.lo.cc.u32 %0, %1, %2, %0;" : "+r"(u_i_m[1]) : "r"(m_loc[1]), "r"(u_i[0]));\
    asm("addc.u32      %0, %0,  0    ;" : "+r"(carry));\
    asm("add.u32       %0, %1,  0    ;" : "=r"(u_i_m[2]) : "r"(carry));\
    asm("add.u32       %0,  0,  0    ;" : "=r"(carry));\
    asm("mad.hi.cc.u32 %0, %1, %2, %0;" : "+r"(u_i_m[2]) : "r"(m_loc[1]), "r"(u_i[0]));\
    asm("addc.u32      %0, %0,  0    ;" : "+r"(carry));\
    asm("mad.lo.cc.u32 %0, %1, %2, %0;" : "+r"(u_i_m[2]) : "r"(m_loc[2]), "r"(u_i[0]));\
    asm("addc.u32      %0, %0,  0    ;" : "+r"(carry));\
    asm("add.u32       %0, %1,  0    ;" : "=r"(u_i_m[3]) : "r"(carry));\
    asm("add.u32       %0,  0,  0    ;" : "=r"(carry));\
    asm("mad.hi.cc.u32 %0, %1, %2, %0;" : "+r"(u_i_m[3]) : "r"(m_loc[2]), "r"(u_i[0]));\
    asm("addc.u32      %0, %0,  0    ;" : "+r"(carry));\
    asm("mad.lo.cc.u32 %0, %1, %2, %0;" : "+r"(u_i_m[3]) : "r"(m_loc[3]), "r"(u_i[0]));\
    asm("addc.u32      %0, %0,  0    ;" : "+r"(carry));\
    asm("mad.hi.u32    %0, %1, %2, %3;" : "=r"(u_i_m[4]) : "r"(m_loc[3]), "r"(u_i[0]), "r"(carry));\
    }\
    asm("add.cc.u32  %0, %1, %2;" : "=r"(A[2]) : "r"(A[2]), "r"(u_i_m[0]));\
    asm("addc.cc.u32 %0, %1, %2;" : "=r"(A[3]) : "r"(A[3]), "r"(u_i_m[1]));\
    asm("addc.cc.u32 %0, %1, %2;" : "=r"(A[4]) : "r"(A[4]), "r"(u_i_m[2]));\
    asm("addc.cc.u32 %0, %1, %2;" : "=r"(A[5]) : "r"(A[5]), "r"(u_i_m[3]));\
    asm("addc.cc.u32 %0, %1, %2;" : "=r"(A[6]) : "r"(A[6]), "r"(u_i_m[4]));\
    asm("addc.u32    %0, %1,  0;" : "=r"(A[7]) : "r"(A[7]));\
    asm("mul.lo.u32 %0, %1, %2;" : "=r"(u_i[0]) : "r"(A[3]), "r"(m_prime));\
    {\
    uint32_t carry = 0;\
    asm("mul.lo.u32    %0, %1, %2    ;" : "=r"(u_i_m[0]) : "r"(m_loc[0]), "r"(u_i[0]));\
    asm("mul.hi.u32    %0, %1, %2    ;" : "=r"(u_i_m[1]) : "r"(m_loc[0]), "r"(u_i[0]));\
    asm("mad.lo.cc.u32 %0, %1, %2, %0;" : "+r"(u_i_m[1]) : "r"(m_loc[1]), "r"(u_i[0]));\
    asm("addc.u32      %0, %0,  0    ;" : "+r"(carry));\
    asm("add.u32       %0, %1,  0    ;" : "=r"(u_i_m[2]) : "r"(carry));\
    asm("add.u32       %0,  0,  0    ;" : "=r"(carry));\
    asm("mad.hi.cc.u32 %0, %1, %2, %0;" : "+r"(u_i_m[2]) : "r"(m_loc[1]), "r"(u_i[0]));\
    asm("addc.u32      %0, %0,  0    ;" : "+r"(carry));\
    asm("mad.lo.cc.u32 %0, %1, %2, %0;" : "+r"(u_i_m[2]) : "r"(m_loc[2]), "r"(u_i[0]));\
    asm("addc.u32      %0, %0,  0    ;" : "+r"(carry));\
    asm("add.u32       %0, %1,  0    ;" : "=r"(u_i_m[3]) : "r"(carry));\
    asm("add.u32       %0,  0,  0    ;" : "=r"(carry));\
    asm("mad.hi.cc.u32 %0, %1, %2, %0;" : "+r"(u_i_m[3]) : "r"(m_loc[2]), "r"(u_i[0]));\
    asm("addc.u32      %0, %0,  0    ;" : "+r"(carry));\
    asm("mad.lo.cc.u32 %0, %1, %2, %0;" : "+r"(u_i_m[3]) : "r"(m_loc[3]), "r"(u_i[0]));\
    asm("addc.u32      %0, %0,  0    ;" : "+r"(carry));\
    asm("mad.hi.u32    %0, %1, %2, %3;" : "=r"(u_i_m[4]) : "r"(m_loc[3]), "r"(u_i[0]), "r"(carry));\
    }\
    asm("add.cc.u32  %0, %1, %2;" : "=r"(A[3]) : "r"(A[3]), "r"(u_i_m[0]));\
    asm("addc.cc.u32 %0, %1, %2;" : "=r"(A[4]) : "r"(A[4]), "r"(u_i_m[1]));\
    asm("addc.cc.u32 %0, %1, %2;" : "=r"(A[5]) : "r"(A[5]), "r"(u_i_m[2]));\
    asm("addc.cc.u32 %0, %1, %2;" : "=r"(A[6]) : "r"(A[6]), "r"(u_i_m[3]));\
    asm("addc.u32    %0, %1, %2;" : "=r"(A[7]) : "r"(A[7]), "r"(u_i_m[4]));\
    uint32_t upper = 0;\
    asm("and.b32 %0, %0, 0xffffe000;" : "+r"(A[3]) : );\
    asm("shr.b32 %0, %0, 13;" : "+r"(A[3]) : );\
    asm("and.b32 %0, %1, 0x00001fff;" : "=r"(upper) : "r"(A[4]));\
    asm("shl.b32 %0, %0, 19;" : "+r"(upper) : );\
    asm("or.b32  %0, %0, %1;" : "+r"(A[3]) : "r"(upper));\
    asm("and.b32 %0, %0, 0xffffe000;" : "+r"(A[4]) : );\
    asm("shr.b32 %0, %0, 13;" : "+r"(A[4]) : );\
    asm("and.b32 %0, %1, 0x00001fff;" : "=r"(upper) : "r"(A[5]));\
    asm("shl.b32 %0, %0, 19;" : "+r"(upper) : );\
    asm("or.b32  %0, %0, %1;" : "+r"(A[4]) : "r"(upper));\
    asm("and.b32 %0, %0, 0xffffe000;" : "+r"(A[5]) : );\
    asm("shr.b32 %0, %0, 13;" : "+r"(A[5]) : );\
    asm("and.b32 %0, %1, 0x00001fff;" : "=r"(upper) : "r"(A[6]));\
    asm("shl.b32 %0, %0, 19;" : "+r"(upper) : );\
    asm("or.b32  %0, %0, %1;" : "+r"(A[5]) : "r"(upper));\
    asm("and.b32 %0, %0, 0xffffe000;" : "+r"(A[6]) : );\
    asm("shr.b32 %0, %0, 13;" : "+r"(A[6]) : );\
    asm("and.b32 %0, %1, 0x00001fff;" : "=r"(upper) : "r"(A[7]));\
    asm("shl.b32 %0, %0, 19;" : "+r"(upper) : );\
    asm("or.b32  %0, %0, %1;" : "+r"(A[6]) : "r"(upper));\
    asm("and.b32 %0, %0, 0xffffe000;" : "+r"(A[7]) : );\
    asm("shr.b32 %0, %0, 13;" : "+r"(A[7]) : );\
    uint32_t mask[8] = {0, 0, 0, 0, 0, 0, 0, 0};\
    asm("sub.cc.u32  %0, %1, %2;" : "=r"(A[0]) : "r"(A[0]), "r"(m_loc[0]));\
    asm("subc.cc.u32 %0, %1, %2;" : "=r"(A[1]) : "r"(A[1]), "r"(m_loc[1]));\
    asm("subc.cc.u32 %0, %1, %2;" : "=r"(A[2]) : "r"(A[2]), "r"(m_loc[2]));\
    asm("subc.cc.u32 %0, %1, %2;" : "=r"(A[3]) : "r"(A[3]), "r"(m_loc[3]));\
    asm("subc.cc.u32 %0, %1,  0;" : "=r"(A[4]) : "r"(A[4]));\
    asm("subc.cc.u32 %0, %1,  0;" : "=r"(A[5]) : "r"(A[5]));\
    asm("subc.cc.u32 %0, %1,  0;" : "=r"(A[6]) : "r"(A[6]));\
    asm("subc.cc.u32 %0, %1,  0;" : "=r"(A[7]) : "r"(A[7]));\
    asm("subc.cc.u32 %0, %1, %2;" : "=r"(mask[0]) : "r"(mask[0]), "r"(mask[0]));\
    asm("subc.cc.u32 %0, %1, %2;" : "=r"(mask[1]) : "r"(mask[1]), "r"(mask[1]));\
    asm("subc.cc.u32 %0, %1, %2;" : "=r"(mask[2]) : "r"(mask[2]), "r"(mask[2]));\
    asm("subc.cc.u32 %0, %1, %2;" : "=r"(mask[3]) : "r"(mask[3]), "r"(mask[3]));\
    asm("subc.cc.u32 %0, %1, %2;" : "=r"(mask[4]) : "r"(mask[4]), "r"(mask[4]));\
    asm("subc.cc.u32 %0, %1, %2;" : "=r"(mask[5]) : "r"(mask[5]), "r"(mask[5]));\
    asm("subc.cc.u32 %0, %1, %2;" : "=r"(mask[6]) : "r"(mask[6]), "r"(mask[6]));\
    asm("subc.u32    %0, %1, %2;" : "=r"(mask[7]) : "r"(mask[7]), "r"(mask[7]));\
    asm("and.b32     %0, %1, %2;" : "=r"(mask[0]) : "r"(mask[0]), "r"(m_loc[0]));\
    asm("and.b32     %0, %1, %2;" : "=r"(mask[1]) : "r"(mask[1]), "r"(m_loc[1]));\
    asm("and.b32     %0, %1, %2;" : "=r"(mask[2]) : "r"(mask[2]), "r"(m_loc[2]));\
    asm("and.b32     %0, %1, %2;" : "=r"(mask[3]) : "r"(mask[3]), "r"(m_loc[3]));\
    asm("and.b32     %0, %1,  0;" : "=r"(mask[4]) : "r"(mask[4]));\
    asm("and.b32     %0, %1,  0;" : "=r"(mask[5]) : "r"(mask[5]));\
    asm("and.b32     %0, %1,  0;" : "=r"(mask[6]) : "r"(mask[6]));\
    asm("and.b32     %0, %1,  0;" : "=r"(mask[7]) : "r"(mask[7]));\
    asm("add.cc.u32  %0, %1, %2;" : "=r"(A[0]) : "r"(A[0]), "r"(mask[0]));\
    asm("addc.cc.u32 %0, %1, %2;" : "=r"(A[1]) : "r"(A[1]), "r"(mask[1]));\
    asm("addc.cc.u32 %0, %1, %2;" : "=r"(A[2]) : "r"(A[2]), "r"(mask[2]));\
    asm("addc.cc.u32 %0, %1, %2;" : "=r"(A[3]) : "r"(A[3]), "r"(mask[3]));\
    asm("addc.cc.u32 %0, %1, %2;" : "=r"(A[4]) : "r"(A[4]), "r"(mask[4]));\
    asm("addc.cc.u32 %0, %1, %2;" : "=r"(A[5]) : "r"(A[5]), "r"(mask[5]));\
    asm("addc.cc.u32 %0, %1, %2;" : "=r"(A[6]) : "r"(A[6]), "r"(mask[6]));\
    asm("addc.u32    %0, %1, %2;" : "=r"(A[7]) : "r"(A[7]), "r"(mask[7]));\
    asm("add.u32 %0, %1,  0;" : "=r"(c_loc[0]) : "r"(A[3]));\
    asm("add.u32 %0, %1,  0;" : "=r"(c_loc[1]) : "r"(A[4]));\
    asm("add.u32 %0, %1,  0;" : "=r"(c_loc[2]) : "r"(A[5]));\
    asm("add.u32 %0, %1,  0;" : "=r"(c_loc[3]) : "r"(A[6]));\
}
