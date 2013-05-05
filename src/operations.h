#include "bignum_types.h"


// Example of the schoolbook addition algorithm we will use if bignums were
// represented on 5 words:
//
//                                      A[4]---A[3]---A[2]---A[1]---A[0]
//                                    + B[4]---B[3]---B[2]---B[1]---B[0]
// -----------------------------------------------------------------------
// |      |      |      |      |      | A[4] | A[3] | A[2] | A[1] | A[0] |
// |      |      |      |      |      |  +   |  +   |  +   |  +   |  +   |
// |      |      |      |      |      | B[4] | B[3] | B[2] | B[1] | B[0] |
// |      |      |      |      |      |  +   |  +   |  +   |  +   |      |
// |      |      |      |      |      |carry |carry |carry |carry |      |
// -----------------------------------------------------------------------
// |   0  |   0  |   0  |   0  | C[5] | C[4] | C[3] | C[2] | C[1] | C[0] |
//
// Note: it is possible that C[5] is also 0 if we are sure that the addition of
// 2 bignums will never require more words than the current number the bignums
// have.
#define add_loc(c_loc, a_loc, b_loc) {\
    asm("add.cc.u32  %0, %1, %2;" : "=r"(c_loc[0]) : "r"(a_loc[0]), "r"(b_loc[0]));\
    asm("addc.cc.u32 %0, %1, %2;" : "=r"(c_loc[1]) : "r"(a_loc[1]), "r"(b_loc[1]));\
    asm("addc.cc.u32 %0, %1, %2;" : "=r"(c_loc[2]) : "r"(a_loc[2]), "r"(b_loc[2]));\
    asm("addc.cc.u32 %0, %1, %2;" : "=r"(c_loc[3]) : "r"(a_loc[3]), "r"(b_loc[3]));\
    asm("addc.cc.u32 %0, %1, %2;" : "=r"(c_loc[4]) : "r"(a_loc[4]), "r"(b_loc[4]));\
    asm("addc.cc.u32 %0, %1, %2;" : "=r"(c_loc[5]) : "r"(a_loc[5]), "r"(b_loc[5]));\
    asm("addc.cc.u32 %0, %1, %2;" : "=r"(c_loc[6]) : "r"(a_loc[6]), "r"(b_loc[6]));\
    asm("addc.cc.u32 %0, %1, %2;" : "=r"(c_loc[7]) : "r"(a_loc[7]), "r"(b_loc[7]));\
    asm("addc.u32    %0, %1, %2;" : "=r"(c_loc[8]) : "r"(a_loc[8]), "r"(b_loc[8]));\
}

#define add_glo(c_glo, a_glo, b_glo, tid) {\
    asm("add.cc.u32  %0, %1, %2;" : "=r"(c_glo[COAL_IDX(0, tid)]) : "r"(a_glo[COAL_IDX(0, tid)]), "r"(b_glo[COAL_IDX(0, tid)]));\
    asm("addc.cc.u32 %0, %1, %2;" : "=r"(c_glo[COAL_IDX(1, tid)]) : "r"(a_glo[COAL_IDX(1, tid)]), "r"(b_glo[COAL_IDX(1, tid)]));\
    asm("addc.cc.u32 %0, %1, %2;" : "=r"(c_glo[COAL_IDX(2, tid)]) : "r"(a_glo[COAL_IDX(2, tid)]), "r"(b_glo[COAL_IDX(2, tid)]));\
    asm("addc.cc.u32 %0, %1, %2;" : "=r"(c_glo[COAL_IDX(3, tid)]) : "r"(a_glo[COAL_IDX(3, tid)]), "r"(b_glo[COAL_IDX(3, tid)]));\
    asm("addc.cc.u32 %0, %1, %2;" : "=r"(c_glo[COAL_IDX(4, tid)]) : "r"(a_glo[COAL_IDX(4, tid)]), "r"(b_glo[COAL_IDX(4, tid)]));\
    asm("addc.cc.u32 %0, %1, %2;" : "=r"(c_glo[COAL_IDX(5, tid)]) : "r"(a_glo[COAL_IDX(5, tid)]), "r"(b_glo[COAL_IDX(5, tid)]));\
    asm("addc.cc.u32 %0, %1, %2;" : "=r"(c_glo[COAL_IDX(6, tid)]) : "r"(a_glo[COAL_IDX(6, tid)]), "r"(b_glo[COAL_IDX(6, tid)]));\
    asm("addc.cc.u32 %0, %1, %2;" : "=r"(c_glo[COAL_IDX(7, tid)]) : "r"(a_glo[COAL_IDX(7, tid)]), "r"(b_glo[COAL_IDX(7, tid)]));\
    asm("addc.u32    %0, %1, %2;" : "=r"(c_glo[COAL_IDX(8, tid)]) : "r"(a_glo[COAL_IDX(8, tid)]), "r"(b_glo[COAL_IDX(8, tid)]));\
}


// Example of the schoolbook subtraction algorithm we will use if bignums were
// represented on 5 words:
//
//                                      A[4]---A[3]---A[2]---A[1]---A[0]
//                                    + B[4]---B[3]---B[2]---B[1]---B[0]
// -----------------------------------------------------------------------
// |   0  |   0  |   0  |   0  |   0  | A[4] | A[3] | A[2] | A[1] | A[0] |
// |   -  |   -  |   -  |   -  |   -  |  -   |  -   |  -   |  -   |  -   |
// |borrow|borrow|borrow|borrow|borrow| B[4] | B[3] | B[2] | B[1] | B[0] |
// |      |      |      |      |      |  -   |  -   |  -   |  -   |      |
// |      |      |      |      |      |borrow|borrow|borrow|borrow|      |
// -----------------------------------------------------------------------
// |0/11..|0/11..|0/11..|0/11..| C[5] | C[4] | C[3] | C[2] | C[1] | C[0] |
//
// Note: it is possible that C[5] is also 0/11.. if we are sure that the
// addition of 2 bignums will never require more words than the current number
// the bignums have.
#define sub_loc(c_loc, a_loc, b_loc) {\
    asm("sub.cc.u32  %0, %1, %2;" : "=r"(c_loc[0]) : "r"(a_loc[0]), "r"(b_loc[0]));\
    asm("subc.cc.u32 %0, %1, %2;" : "=r"(c_loc[1]) : "r"(a_loc[1]), "r"(b_loc[1]));\
    asm("subc.cc.u32 %0, %1, %2;" : "=r"(c_loc[2]) : "r"(a_loc[2]), "r"(b_loc[2]));\
    asm("subc.cc.u32 %0, %1, %2;" : "=r"(c_loc[3]) : "r"(a_loc[3]), "r"(b_loc[3]));\
    asm("subc.cc.u32 %0, %1, %2;" : "=r"(c_loc[4]) : "r"(a_loc[4]), "r"(b_loc[4]));\
    asm("subc.cc.u32 %0, %1, %2;" : "=r"(c_loc[5]) : "r"(a_loc[5]), "r"(b_loc[5]));\
    asm("subc.cc.u32 %0, %1, %2;" : "=r"(c_loc[6]) : "r"(a_loc[6]), "r"(b_loc[6]));\
    asm("subc.cc.u32 %0, %1, %2;" : "=r"(c_loc[7]) : "r"(a_loc[7]), "r"(b_loc[7]));\
    asm("subc.u32    %0, %1, %2;" : "=r"(c_loc[8]) : "r"(a_loc[8]), "r"(b_loc[8]));\
}

#define sub_glo(c_glo, a_glo, b_glo, tid) {\
    asm("sub.cc.u32  %0, %1, %2;" : "=r"(c_glo[COAL_IDX(0, tid)]) : "r"(a_glo[COAL_IDX(0, tid)]), "r"(b_glo[COAL_IDX(0, tid)]));\
    asm("subc.cc.u32 %0, %1, %2;" : "=r"(c_glo[COAL_IDX(1, tid)]) : "r"(a_glo[COAL_IDX(1, tid)]), "r"(b_glo[COAL_IDX(1, tid)]));\
    asm("subc.cc.u32 %0, %1, %2;" : "=r"(c_glo[COAL_IDX(2, tid)]) : "r"(a_glo[COAL_IDX(2, tid)]), "r"(b_glo[COAL_IDX(2, tid)]));\
    asm("subc.cc.u32 %0, %1, %2;" : "=r"(c_glo[COAL_IDX(3, tid)]) : "r"(a_glo[COAL_IDX(3, tid)]), "r"(b_glo[COAL_IDX(3, tid)]));\
    asm("subc.cc.u32 %0, %1, %2;" : "=r"(c_glo[COAL_IDX(4, tid)]) : "r"(a_glo[COAL_IDX(4, tid)]), "r"(b_glo[COAL_IDX(4, tid)]));\
    asm("subc.cc.u32 %0, %1, %2;" : "=r"(c_glo[COAL_IDX(5, tid)]) : "r"(a_glo[COAL_IDX(5, tid)]), "r"(b_glo[COAL_IDX(5, tid)]));\
    asm("subc.cc.u32 %0, %1, %2;" : "=r"(c_glo[COAL_IDX(6, tid)]) : "r"(a_glo[COAL_IDX(6, tid)]), "r"(b_glo[COAL_IDX(6, tid)]));\
    asm("subc.cc.u32 %0, %1, %2;" : "=r"(c_glo[COAL_IDX(7, tid)]) : "r"(a_glo[COAL_IDX(7, tid)]), "r"(b_glo[COAL_IDX(7, tid)]));\
    asm("subc.u32    %0, %1, %2;" : "=r"(c_glo[COAL_IDX(8, tid)]) : "r"(a_glo[COAL_IDX(8, tid)]), "r"(b_glo[COAL_IDX(8, tid)]));\
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
#define mul_loc(c_loc, a_loc, b_loc) {\
    uint32_t carry = 0;
    asm("mul.lo.u32    %0    , %1    , %2    ;" : "=r"(c_loc[0]) : "r"(a_loc[0]), "r"(b_loc[0]));\
    asm("add.u32       %carry,  0    ,  0    ;" :                :                             );\
    asm("mul.hi.u32    %0    , %1    , %2    ;" : "=r"(c_loc[1]) : "r"(b_loc[0]), "r"(a_loc[0]));\
    asm("mad.lo.cc.u32 %0    , %1    , %2, %0;" : "+r"(c_loc[1]) : "r"(b_loc[0]), "r"(a_loc[1]));\
    asm("addc.u32      %carry, %carry,  0    ;" :                :                             );\
    asm("mad.lo.cc.u32 %0    , %1    , %2, %0;" : "+r"(c_loc[1]) : "r"(b_loc[1]), "r"(a_loc[0]));\
    asm("addc.u32      %carry, %carry,  0    ;" :                :                             );\
    asm("add.u32       %0    , %carry,  0    ;" : "=r"(c_loc[2]) :                             );\
    asm("add.u32       %carry,  0    ,  0    ;" :                :                             );\
    asm("mad.hi.cc.u32 %0    , %1    , %2, %0;" : "+r"(c_loc[2]) : "r"(b_loc[0]), "r"(a_loc[1]));\
    asm("addc.u32      %carry, %carry,  0    ;" :                :                             );\
    asm("mad.hi.cc.u32 %0    , %1    , %2, %0;" : "+r"(c_loc[2]) : "r"(b_loc[1]), "r"(a_loc[0]));\
    asm("addc.u32      %carry, %carry,  0    ;" :                :                             );\
    asm("mad.lo.cc.u32 %0    , %1    , %2, %0;" : "+r"(c_loc[2]) : "r"(b_loc[0]), "r"(a_loc[2]));\
    asm("addc.u32      %carry, %carry,  0    ;" :                :                             );\
    asm("mad.lo.cc.u32 %0    , %1    , %2, %0;" : "+r"(c_loc[2]) : "r"(b_loc[1]), "r"(a_loc[1]));\
    asm("addc.u32      %carry, %carry,  0    ;" :                :                             );\
    asm("mad.lo.cc.u32 %0    , %1    , %2, %0;" : "+r"(c_loc[2]) : "r"(b_loc[2]), "r"(a_loc[0]));\
    asm("addc.u32      %carry, %carry,  0    ;" :                :                             );\
    asm("add.u32       %0    , %carry,  0    ;" : "=r"(c_loc[3]) :                             );\
    asm("add.u32       %carry,  0    ,  0    ;" :                :                             );\
    asm("mad.hi.cc.u32 %0    , %1    , %2, %0;" : "+r"(c_loc[3]) : "r"(b_loc[0]), "r"(a_loc[2]));\
    asm("addc.u32      %carry, %carry,  0    ;" :                :                             );\
    asm("mad.hi.cc.u32 %0    , %1    , %2, %0;" : "+r"(c_loc[3]) : "r"(b_loc[1]), "r"(a_loc[1]));\
    asm("addc.u32      %carry, %carry,  0    ;" :                :                             );\
    asm("mad.hi.cc.u32 %0    , %1    , %2, %0;" : "+r"(c_loc[3]) : "r"(b_loc[2]), "r"(a_loc[0]));\
    asm("addc.u32      %carry, %carry,  0    ;" :                :                             );\
    asm("mad.lo.cc.u32 %0    , %1    , %2, %0;" : "+r"(c_loc[3]) : "r"(b_loc[0]), "r"(a_loc[3]));\
    asm("addc.u32      %carry, %carry,  0    ;" :                :                             );\
    asm("mad.lo.cc.u32 %0    , %1    , %2, %0;" : "+r"(c_loc[3]) : "r"(b_loc[1]), "r"(a_loc[2]));\
    asm("addc.u32      %carry, %carry,  0    ;" :                :                             );\
    asm("mad.lo.cc.u32 %0    , %1    , %2, %0;" : "+r"(c_loc[3]) : "r"(b_loc[2]), "r"(a_loc[1]));\
    asm("addc.u32      %carry, %carry,  0    ;" :                :                             );\
    asm("mad.lo.cc.u32 %0    , %1    , %2, %0;" : "+r"(c_loc[3]) : "r"(b_loc[3]), "r"(a_loc[0]));\
    asm("addc.u32      %carry, %carry,  0    ;" :                :                             );\
    asm("add.u32       %0    , %carry,  0    ;" : "=r"(c_loc[4]) :                             );\
    asm("add.u32       %carry,  0    ,  0    ;" :                :                             );\
    asm("mad.hi.cc.u32 %0    , %1    , %2, %0;" : "+r"(c_loc[4]) : "r"(b_loc[0]), "r"(a_loc[3]));\
    asm("addc.u32      %carry, %carry,  0    ;" :                :                             );\
    asm("mad.hi.cc.u32 %0    , %1    , %2, %0;" : "+r"(c_loc[4]) : "r"(b_loc[1]), "r"(a_loc[2]));\
    asm("addc.u32      %carry, %carry,  0    ;" :                :                             );\
    asm("mad.hi.cc.u32 %0    , %1    , %2, %0;" : "+r"(c_loc[4]) : "r"(b_loc[2]), "r"(a_loc[1]));\
    asm("addc.u32      %carry, %carry,  0    ;" :                :                             );\
    asm("mad.hi.cc.u32 %0    , %1    , %2, %0;" : "+r"(c_loc[4]) : "r"(b_loc[3]), "r"(a_loc[0]));\
    asm("addc.u32      %carry, %carry,  0    ;" :                :                             );\
    asm("mad.lo.cc.u32 %0    , %1    , %2, %0;" : "+r"(c_loc[4]) : "r"(b_loc[0]), "r"(a_loc[4]));\
    asm("addc.u32      %carry, %carry,  0    ;" :                :                             );\
    asm("mad.lo.cc.u32 %0    , %1    , %2, %0;" : "+r"(c_loc[4]) : "r"(b_loc[1]), "r"(a_loc[3]));\
    asm("addc.u32      %carry, %carry,  0    ;" :                :                             );\
    asm("mad.lo.cc.u32 %0    , %1    , %2, %0;" : "+r"(c_loc[4]) : "r"(b_loc[2]), "r"(a_loc[2]));\
    asm("addc.u32      %carry, %carry,  0    ;" :                :                             );\
    asm("mad.lo.cc.u32 %0    , %1    , %2, %0;" : "+r"(c_loc[4]) : "r"(b_loc[3]), "r"(a_loc[1]));\
    asm("addc.u32      %carry, %carry,  0    ;" :                :                             );\
    asm("mad.lo.cc.u32 %0    , %1    , %2, %0;" : "+r"(c_loc[4]) : "r"(b_loc[4]), "r"(a_loc[0]));\
    asm("addc.u32      %carry, %carry,  0    ;" :                :                             );\
    asm("add.u32       %0    , %carry,  0    ;" : "=r"(c_loc[5]) :                             );\
    asm("add.u32       %carry,  0    ,  0    ;" :                :                             );\
    asm("mad.hi.cc.u32 %0    , %1    , %2, %0;" : "+r"(c_loc[5]) : "r"(b_loc[0]), "r"(a_loc[4]));\
    asm("addc.u32      %carry, %carry,  0    ;" :                :                             );\
    asm("mad.hi.cc.u32 %0    , %1    , %2, %0;" : "+r"(c_loc[5]) : "r"(b_loc[1]), "r"(a_loc[3]));\
    asm("addc.u32      %carry, %carry,  0    ;" :                :                             );\
    asm("mad.hi.cc.u32 %0    , %1    , %2, %0;" : "+r"(c_loc[5]) : "r"(b_loc[2]), "r"(a_loc[2]));\
    asm("addc.u32      %carry, %carry,  0    ;" :                :                             );\
    asm("mad.hi.cc.u32 %0    , %1    , %2, %0;" : "+r"(c_loc[5]) : "r"(b_loc[3]), "r"(a_loc[1]));\
    asm("addc.u32      %carry, %carry,  0    ;" :                :                             );\
    asm("mad.hi.cc.u32 %0    , %1    , %2, %0;" : "+r"(c_loc[5]) : "r"(b_loc[4]), "r"(a_loc[0]));\
    asm("addc.u32      %carry, %carry,  0    ;" :                :                             );\
    asm("mad.lo.cc.u32 %0    , %1    , %2, %0;" : "+r"(c_loc[5]) : "r"(b_loc[1]), "r"(a_loc[4]));\
    asm("addc.u32      %carry, %carry,  0    ;" :                :                             );\
    asm("mad.lo.cc.u32 %0    , %1    , %2, %0;" : "+r"(c_loc[5]) : "r"(b_loc[2]), "r"(a_loc[3]));\
    asm("addc.u32      %carry, %carry,  0    ;" :                :                             );\
    asm("mad.lo.cc.u32 %0    , %1    , %2, %0;" : "+r"(c_loc[5]) : "r"(b_loc[3]), "r"(a_loc[2]));\
    asm("addc.u32      %carry, %carry,  0    ;" :                :                             );\
    asm("mad.lo.cc.u32 %0    , %1    , %2, %0;" : "+r"(c_loc[5]) : "r"(b_loc[4]), "r"(a_loc[1]));\
    asm("addc.u32      %carry, %carry,  0    ;" :                :                             );\
    asm("add.u32       %0    , %carry,  0    ;" : "=r"(c_loc[6]) :                             );\
    asm("add.u32       %carry,  0    ,  0    ;" :                :                             );\
    asm("mad.hi.cc.u32 %0    , %1    , %2, %0;" : "+r"(c_loc[6]) : "r"(b_loc[1]), "r"(a_loc[4]));\
    asm("addc.u32      %carry, %carry,  0    ;" :                :                             );\
    asm("mad.hi.cc.u32 %0    , %1    , %2, %0;" : "+r"(c_loc[6]) : "r"(b_loc[2]), "r"(a_loc[3]));\
    asm("addc.u32      %carry, %carry,  0    ;" :                :                             );\
    asm("mad.hi.cc.u32 %0    , %1    , %2, %0;" : "+r"(c_loc[6]) : "r"(b_loc[3]), "r"(a_loc[2]));\
    asm("addc.u32      %carry, %carry,  0    ;" :                :                             );\
    asm("mad.hi.cc.u32 %0    , %1    , %2, %0;" : "+r"(c_loc[6]) : "r"(b_loc[4]), "r"(a_loc[1]));\
    asm("addc.u32      %carry, %carry,  0    ;" :                :                             );\
    asm("mad.lo.cc.u32 %0    , %1    , %2, %0;" : "+r"(c_loc[6]) : "r"(b_loc[2]), "r"(a_loc[4]));\
    asm("addc.u32      %carry, %carry,  0    ;" :                :                             );\
    asm("mad.lo.cc.u32 %0    , %1    , %2, %0;" : "+r"(c_loc[6]) : "r"(b_loc[3]), "r"(a_loc[3]));\
    asm("addc.u32      %carry, %carry,  0    ;" :                :                             );\
    asm("mad.lo.cc.u32 %0    , %1    , %2, %0;" : "+r"(c_loc[6]) : "r"(b_loc[4]), "r"(a_loc[2]));\
    asm("addc.u32      %carry, %carry,  0    ;" :                :                             );\
    asm("add.u32       %0    , %carry,  0    ;" : "=r"(c_loc[7]) :                             );\
    asm("add.u32       %carry,  0    ,  0    ;" :                :                             );\
    asm("mad.hi.cc.u32 %0    , %1    , %2, %0;" : "+r"(c_loc[7]) : "r"(b_loc[2]), "r"(a_loc[4]));\
    asm("addc.u32      %carry, %carry,  0    ;" :                :                             );\
    asm("mad.hi.cc.u32 %0    , %1    , %2, %0;" : "+r"(c_loc[7]) : "r"(b_loc[3]), "r"(a_loc[3]));\
    asm("addc.u32      %carry, %carry,  0    ;" :                :                             );\
    asm("mad.hi.cc.u32 %0    , %1    , %2, %0;" : "+r"(c_loc[7]) : "r"(b_loc[4]), "r"(a_loc[2]));\
    asm("addc.u32      %carry, %carry,  0    ;" :                :                             );\
    asm("mad.lo.cc.u32 %0    , %1    , %2, %0;" : "+r"(c_loc[7]) : "r"(b_loc[3]), "r"(a_loc[4]));\
    asm("addc.u32      %carry, %carry,  0    ;" :                :                             );\
    asm("mad.lo.cc.u32 %0    , %1    , %2, %0;" : "+r"(c_loc[7]) : "r"(b_loc[4]), "r"(a_loc[3]));\
    asm("addc.u32      %carry, %carry,  0    ;" :                :                             );\
    asm("add.u32       %0    , %carry,  0    ;" : "=r"(c_loc[8]) :                             );\
    asm("add.u32       %carry,  0    ,  0    ;" :                :                             );\
    asm("mad.hi.cc.u32 %0    , %1    , %2, %0;" : "+r"(c_loc[8]) : "r"(b_loc[3]), "r"(a_loc[4]));\
    asm("addc.u32      %carry, %carry,  0    ;" :                :                             );\
    asm("mad.hi.cc.u32 %0    , %1    , %2, %0;" : "+r"(c_loc[8]) : "r"(b_loc[4]), "r"(a_loc[3]));\
    asm("addc.u32      %carry, %carry,  0    ;" :                :                             );\
    asm("mad.lo.cc.u32 %0    , %1    , %2, %0;" : "+r"(c_loc[8]) : "r"(b_loc[4]), "r"(a_loc[4]));\
}

#define mul_glo(c_glo, a_glo, b_glo, tid) {\
    uint32_t carry = 0;
    asm("mul.lo.u32    %0    , %1    , %2    ;" : "=r"(c_glo[COAL_IDX(0, tid)]) : "r"(a_glo[COAL_IDX(0, tid)]), "r"(b_glo[COAL_IDX(0, tid)]));\
    asm("add.u32       %carry,  0    ,  0    ;" :                :                             );\
    asm("mul.hi.u32    %0    , %1    , %2    ;" : "=r"(c_glo[COAL_IDX(1, tid)]) : "r"(b_glo[COAL_IDX(0, tid)]), "r"(a_glo[COAL_IDX(0, tid)]));\
    asm("mad.lo.cc.u32 %0    , %1    , %2, %0;" : "+r"(c_glo[COAL_IDX(1, tid)]) : "r"(b_glo[COAL_IDX(0, tid)]), "r"(a_glo[COAL_IDX(1, tid)]));\
    asm("addc.u32      %carry, %carry,  0    ;" :                :                             );\
    asm("mad.lo.cc.u32 %0    , %1    , %2, %0;" : "+r"(c_glo[COAL_IDX(1, tid)]) : "r"(b_glo[COAL_IDX(1, tid)]), "r"(a_glo[COAL_IDX(0, tid)]));\
    asm("addc.u32      %carry, %carry,  0    ;" :                :                             );\
    asm("add.u32       %0    , %carry,  0    ;" : "=r"(c_glo[COAL_IDX(2, tid)]) :                             );\
    asm("add.u32       %carry,  0    ,  0    ;" :                :                             );\
    asm("mad.hi.cc.u32 %0    , %1    , %2, %0;" : "+r"(c_glo[COAL_IDX(2, tid)]) : "r"(b_glo[COAL_IDX(0, tid)]), "r"(a_glo[COAL_IDX(1, tid)]));\
    asm("addc.u32      %carry, %carry,  0    ;" :                :                             );\
    asm("mad.hi.cc.u32 %0    , %1    , %2, %0;" : "+r"(c_glo[COAL_IDX(2, tid)]) : "r"(b_glo[COAL_IDX(1, tid)]), "r"(a_glo[COAL_IDX(0, tid)]));\
    asm("addc.u32      %carry, %carry,  0    ;" :                :                             );\
    asm("mad.lo.cc.u32 %0    , %1    , %2, %0;" : "+r"(c_glo[COAL_IDX(2, tid)]) : "r"(b_glo[COAL_IDX(0, tid)]), "r"(a_glo[COAL_IDX(2, tid)]));\
    asm("addc.u32      %carry, %carry,  0    ;" :                :                             );\
    asm("mad.lo.cc.u32 %0    , %1    , %2, %0;" : "+r"(c_glo[COAL_IDX(2, tid)]) : "r"(b_glo[COAL_IDX(1, tid)]), "r"(a_glo[COAL_IDX(1, tid)]));\
    asm("addc.u32      %carry, %carry,  0    ;" :                :                             );\
    asm("mad.lo.cc.u32 %0    , %1    , %2, %0;" : "+r"(c_glo[COAL_IDX(2, tid)]) : "r"(b_glo[COAL_IDX(2, tid)]), "r"(a_glo[COAL_IDX(0, tid)]));\
    asm("addc.u32      %carry, %carry,  0    ;" :                :                             );\
    asm("add.u32       %0    , %carry,  0    ;" : "=r"(c_glo[COAL_IDX(3, tid)]) :                             );\
    asm("add.u32       %carry,  0    ,  0    ;" :                :                             );\
    asm("mad.hi.cc.u32 %0    , %1    , %2, %0;" : "+r"(c_glo[COAL_IDX(3, tid)]) : "r"(b_glo[COAL_IDX(0, tid)]), "r"(a_glo[COAL_IDX(2, tid)]));\
    asm("addc.u32      %carry, %carry,  0    ;" :                :                             );\
    asm("mad.hi.cc.u32 %0    , %1    , %2, %0;" : "+r"(c_glo[COAL_IDX(3, tid)]) : "r"(b_glo[COAL_IDX(1, tid)]), "r"(a_glo[COAL_IDX(1, tid)]));\
    asm("addc.u32      %carry, %carry,  0    ;" :                :                             );\
    asm("mad.hi.cc.u32 %0    , %1    , %2, %0;" : "+r"(c_glo[COAL_IDX(3, tid)]) : "r"(b_glo[COAL_IDX(2, tid)]), "r"(a_glo[COAL_IDX(0, tid)]));\
    asm("addc.u32      %carry, %carry,  0    ;" :                :                             );\
    asm("mad.lo.cc.u32 %0    , %1    , %2, %0;" : "+r"(c_glo[COAL_IDX(3, tid)]) : "r"(b_glo[COAL_IDX(0, tid)]), "r"(a_glo[COAL_IDX(3, tid)]));\
    asm("addc.u32      %carry, %carry,  0    ;" :                :                             );\
    asm("mad.lo.cc.u32 %0    , %1    , %2, %0;" : "+r"(c_glo[COAL_IDX(3, tid)]) : "r"(b_glo[COAL_IDX(1, tid)]), "r"(a_glo[COAL_IDX(2, tid)]));\
    asm("addc.u32      %carry, %carry,  0    ;" :                :                             );\
    asm("mad.lo.cc.u32 %0    , %1    , %2, %0;" : "+r"(c_glo[COAL_IDX(3, tid)]) : "r"(b_glo[COAL_IDX(2, tid)]), "r"(a_glo[COAL_IDX(1, tid)]));\
    asm("addc.u32      %carry, %carry,  0    ;" :                :                             );\
    asm("mad.lo.cc.u32 %0    , %1    , %2, %0;" : "+r"(c_glo[COAL_IDX(3, tid)]) : "r"(b_glo[COAL_IDX(3, tid)]), "r"(a_glo[COAL_IDX(0, tid)]));\
    asm("addc.u32      %carry, %carry,  0    ;" :                :                             );\
    asm("add.u32       %0    , %carry,  0    ;" : "=r"(c_glo[COAL_IDX(4, tid)]) :                             );\
    asm("add.u32       %carry,  0    ,  0    ;" :                :                             );\
    asm("mad.hi.cc.u32 %0    , %1    , %2, %0;" : "+r"(c_glo[COAL_IDX(4, tid)]) : "r"(b_glo[COAL_IDX(0, tid)]), "r"(a_glo[COAL_IDX(3, tid)]));\
    asm("addc.u32      %carry, %carry,  0    ;" :                :                             );\
    asm("mad.hi.cc.u32 %0    , %1    , %2, %0;" : "+r"(c_glo[COAL_IDX(4, tid)]) : "r"(b_glo[COAL_IDX(1, tid)]), "r"(a_glo[COAL_IDX(2, tid)]));\
    asm("addc.u32      %carry, %carry,  0    ;" :                :                             );\
    asm("mad.hi.cc.u32 %0    , %1    , %2, %0;" : "+r"(c_glo[COAL_IDX(4, tid)]) : "r"(b_glo[COAL_IDX(2, tid)]), "r"(a_glo[COAL_IDX(1, tid)]));\
    asm("addc.u32      %carry, %carry,  0    ;" :                :                             );\
    asm("mad.hi.cc.u32 %0    , %1    , %2, %0;" : "+r"(c_glo[COAL_IDX(4, tid)]) : "r"(b_glo[COAL_IDX(3, tid)]), "r"(a_glo[COAL_IDX(0, tid)]));\
    asm("addc.u32      %carry, %carry,  0    ;" :                :                             );\
    asm("mad.lo.cc.u32 %0    , %1    , %2, %0;" : "+r"(c_glo[COAL_IDX(4, tid)]) : "r"(b_glo[COAL_IDX(0, tid)]), "r"(a_glo[COAL_IDX(4, tid)]));\
    asm("addc.u32      %carry, %carry,  0    ;" :                :                             );\
    asm("mad.lo.cc.u32 %0    , %1    , %2, %0;" : "+r"(c_glo[COAL_IDX(4, tid)]) : "r"(b_glo[COAL_IDX(1, tid)]), "r"(a_glo[COAL_IDX(3, tid)]));\
    asm("addc.u32      %carry, %carry,  0    ;" :                :                             );\
    asm("mad.lo.cc.u32 %0    , %1    , %2, %0;" : "+r"(c_glo[COAL_IDX(4, tid)]) : "r"(b_glo[COAL_IDX(2, tid)]), "r"(a_glo[COAL_IDX(2, tid)]));\
    asm("addc.u32      %carry, %carry,  0    ;" :                :                             );\
    asm("mad.lo.cc.u32 %0    , %1    , %2, %0;" : "+r"(c_glo[COAL_IDX(4, tid)]) : "r"(b_glo[COAL_IDX(3, tid)]), "r"(a_glo[COAL_IDX(1, tid)]));\
    asm("addc.u32      %carry, %carry,  0    ;" :                :                             );\
    asm("mad.lo.cc.u32 %0    , %1    , %2, %0;" : "+r"(c_glo[COAL_IDX(4, tid)]) : "r"(b_glo[COAL_IDX(4, tid)]), "r"(a_glo[COAL_IDX(0, tid)]));\
    asm("addc.u32      %carry, %carry,  0    ;" :                :                             );\
    asm("add.u32       %0    , %carry,  0    ;" : "=r"(c_glo[COAL_IDX(5, tid)]) :                             );\
    asm("add.u32       %carry,  0    ,  0    ;" :                :                             );\
    asm("mad.hi.cc.u32 %0    , %1    , %2, %0;" : "+r"(c_glo[COAL_IDX(5, tid)]) : "r"(b_glo[COAL_IDX(0, tid)]), "r"(a_glo[COAL_IDX(4, tid)]));\
    asm("addc.u32      %carry, %carry,  0    ;" :                :                             );\
    asm("mad.hi.cc.u32 %0    , %1    , %2, %0;" : "+r"(c_glo[COAL_IDX(5, tid)]) : "r"(b_glo[COAL_IDX(1, tid)]), "r"(a_glo[COAL_IDX(3, tid)]));\
    asm("addc.u32      %carry, %carry,  0    ;" :                :                             );\
    asm("mad.hi.cc.u32 %0    , %1    , %2, %0;" : "+r"(c_glo[COAL_IDX(5, tid)]) : "r"(b_glo[COAL_IDX(2, tid)]), "r"(a_glo[COAL_IDX(2, tid)]));\
    asm("addc.u32      %carry, %carry,  0    ;" :                :                             );\
    asm("mad.hi.cc.u32 %0    , %1    , %2, %0;" : "+r"(c_glo[COAL_IDX(5, tid)]) : "r"(b_glo[COAL_IDX(3, tid)]), "r"(a_glo[COAL_IDX(1, tid)]));\
    asm("addc.u32      %carry, %carry,  0    ;" :                :                             );\
    asm("mad.hi.cc.u32 %0    , %1    , %2, %0;" : "+r"(c_glo[COAL_IDX(5, tid)]) : "r"(b_glo[COAL_IDX(4, tid)]), "r"(a_glo[COAL_IDX(0, tid)]));\
    asm("addc.u32      %carry, %carry,  0    ;" :                :                             );\
    asm("mad.lo.cc.u32 %0    , %1    , %2, %0;" : "+r"(c_glo[COAL_IDX(5, tid)]) : "r"(b_glo[COAL_IDX(1, tid)]), "r"(a_glo[COAL_IDX(4, tid)]));\
    asm("addc.u32      %carry, %carry,  0    ;" :                :                             );\
    asm("mad.lo.cc.u32 %0    , %1    , %2, %0;" : "+r"(c_glo[COAL_IDX(5, tid)]) : "r"(b_glo[COAL_IDX(2, tid)]), "r"(a_glo[COAL_IDX(3, tid)]));\
    asm("addc.u32      %carry, %carry,  0    ;" :                :                             );\
    asm("mad.lo.cc.u32 %0    , %1    , %2, %0;" : "+r"(c_glo[COAL_IDX(5, tid)]) : "r"(b_glo[COAL_IDX(3, tid)]), "r"(a_glo[COAL_IDX(2, tid)]));\
    asm("addc.u32      %carry, %carry,  0    ;" :                :                             );\
    asm("mad.lo.cc.u32 %0    , %1    , %2, %0;" : "+r"(c_glo[COAL_IDX(5, tid)]) : "r"(b_glo[COAL_IDX(4, tid)]), "r"(a_glo[COAL_IDX(1, tid)]));\
    asm("addc.u32      %carry, %carry,  0    ;" :                :                             );\
    asm("add.u32       %0    , %carry,  0    ;" : "=r"(c_glo[COAL_IDX(6, tid)]) :                             );\
    asm("add.u32       %carry,  0    ,  0    ;" :                :                             );\
    asm("mad.hi.cc.u32 %0    , %1    , %2, %0;" : "+r"(c_glo[COAL_IDX(6, tid)]) : "r"(b_glo[COAL_IDX(1, tid)]), "r"(a_glo[COAL_IDX(4, tid)]));\
    asm("addc.u32      %carry, %carry,  0    ;" :                :                             );\
    asm("mad.hi.cc.u32 %0    , %1    , %2, %0;" : "+r"(c_glo[COAL_IDX(6, tid)]) : "r"(b_glo[COAL_IDX(2, tid)]), "r"(a_glo[COAL_IDX(3, tid)]));\
    asm("addc.u32      %carry, %carry,  0    ;" :                :                             );\
    asm("mad.hi.cc.u32 %0    , %1    , %2, %0;" : "+r"(c_glo[COAL_IDX(6, tid)]) : "r"(b_glo[COAL_IDX(3, tid)]), "r"(a_glo[COAL_IDX(2, tid)]));\
    asm("addc.u32      %carry, %carry,  0    ;" :                :                             );\
    asm("mad.hi.cc.u32 %0    , %1    , %2, %0;" : "+r"(c_glo[COAL_IDX(6, tid)]) : "r"(b_glo[COAL_IDX(4, tid)]), "r"(a_glo[COAL_IDX(1, tid)]));\
    asm("addc.u32      %carry, %carry,  0    ;" :                :                             );\
    asm("mad.lo.cc.u32 %0    , %1    , %2, %0;" : "+r"(c_glo[COAL_IDX(6, tid)]) : "r"(b_glo[COAL_IDX(2, tid)]), "r"(a_glo[COAL_IDX(4, tid)]));\
    asm("addc.u32      %carry, %carry,  0    ;" :                :                             );\
    asm("mad.lo.cc.u32 %0    , %1    , %2, %0;" : "+r"(c_glo[COAL_IDX(6, tid)]) : "r"(b_glo[COAL_IDX(3, tid)]), "r"(a_glo[COAL_IDX(3, tid)]));\
    asm("addc.u32      %carry, %carry,  0    ;" :                :                             );\
    asm("mad.lo.cc.u32 %0    , %1    , %2, %0;" : "+r"(c_glo[COAL_IDX(6, tid)]) : "r"(b_glo[COAL_IDX(4, tid)]), "r"(a_glo[COAL_IDX(2, tid)]));\
    asm("addc.u32      %carry, %carry,  0    ;" :                :                             );\
    asm("add.u32       %0    , %carry,  0    ;" : "=r"(c_glo[COAL_IDX(7, tid)]) :                             );\
    asm("add.u32       %carry,  0    ,  0    ;" :                :                             );\
    asm("mad.hi.cc.u32 %0    , %1    , %2, %0;" : "+r"(c_glo[COAL_IDX(7, tid)]) : "r"(b_glo[COAL_IDX(2, tid)]), "r"(a_glo[COAL_IDX(4, tid)]));\
    asm("addc.u32      %carry, %carry,  0    ;" :                :                             );\
    asm("mad.hi.cc.u32 %0    , %1    , %2, %0;" : "+r"(c_glo[COAL_IDX(7, tid)]) : "r"(b_glo[COAL_IDX(3, tid)]), "r"(a_glo[COAL_IDX(3, tid)]));\
    asm("addc.u32      %carry, %carry,  0    ;" :                :                             );\
    asm("mad.hi.cc.u32 %0    , %1    , %2, %0;" : "+r"(c_glo[COAL_IDX(7, tid)]) : "r"(b_glo[COAL_IDX(4, tid)]), "r"(a_glo[COAL_IDX(2, tid)]));\
    asm("addc.u32      %carry, %carry,  0    ;" :                :                             );\
    asm("mad.lo.cc.u32 %0    , %1    , %2, %0;" : "+r"(c_glo[COAL_IDX(7, tid)]) : "r"(b_glo[COAL_IDX(3, tid)]), "r"(a_glo[COAL_IDX(4, tid)]));\
    asm("addc.u32      %carry, %carry,  0    ;" :                :                             );\
    asm("mad.lo.cc.u32 %0    , %1    , %2, %0;" : "+r"(c_glo[COAL_IDX(7, tid)]) : "r"(b_glo[COAL_IDX(4, tid)]), "r"(a_glo[COAL_IDX(3, tid)]));\
    asm("addc.u32      %carry, %carry,  0    ;" :                :                             );\
    asm("add.u32       %0    , %carry,  0    ;" : "=r"(c_glo[COAL_IDX(8, tid)]) :                             );\
    asm("add.u32       %carry,  0    ,  0    ;" :                :                             );\
    asm("mad.hi.cc.u32 %0    , %1    , %2, %0;" : "+r"(c_glo[COAL_IDX(8, tid)]) : "r"(b_glo[COAL_IDX(3, tid)]), "r"(a_glo[COAL_IDX(4, tid)]));\
    asm("addc.u32      %carry, %carry,  0    ;" :                :                             );\
    asm("mad.hi.cc.u32 %0    , %1    , %2, %0;" : "+r"(c_glo[COAL_IDX(8, tid)]) : "r"(b_glo[COAL_IDX(4, tid)]), "r"(a_glo[COAL_IDX(3, tid)]));\
    asm("addc.u32      %carry, %carry,  0    ;" :                :                             );\
    asm("mad.lo.cc.u32 %0    , %1    , %2, %0;" : "+r"(c_glo[COAL_IDX(8, tid)]) : "r"(b_glo[COAL_IDX(4, tid)]), "r"(a_glo[COAL_IDX(4, tid)]));\
}
