#include "bignum_types.h"

// Example of the schoolbook addition algorithm we will use if bignums
// were represented on 5 words:
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
asm("{"\
"add.cc.u32 %0, %3, %6;"\
"addc.cc.u32 %1, %4, %7;"\
"addc.u32 %2, %5, %8;"\
"}"\
:\
"=r"(c_loc[0]),\
"=r"(c_loc[1]),\
"=r"(c_loc[2])\
:\
"r"(a_loc[0]),\
"r"(a_loc[1]),\
"r"(a_loc[2]),\
"r"(b_loc[0]),\
"r"(b_loc[1]),\
"r"(b_loc[2])\
);\
}

#define add_glo(c_glo, a_glo, b_glo, tid) {\
asm("{"\
"add.cc.u32 %0, %3, %6;"\
"addc.cc.u32 %1, %4, %7;"\
"addc.u32 %2, %5, %8;"\
"}"\
:\
"=r"(c_glo[COAL_IDX(0, tid)]),\
"=r"(c_glo[COAL_IDX(1, tid)]),\
"=r"(c_glo[COAL_IDX(2, tid)])\
:\
"r"(a_glo[COAL_IDX(0, tid)]),\
"r"(a_glo[COAL_IDX(1, tid)]),\
"r"(a_glo[COAL_IDX(2, tid)]),\
"r"(b_glo[COAL_IDX(0, tid)]),\
"r"(b_glo[COAL_IDX(1, tid)]),\
"r"(b_glo[COAL_IDX(2, tid)])\
);\
}

// Example of the schoolbook subtraction algorithm we will use if bignums
// were represented on 5 words:
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
asm("{"\
"sub.cc.u32 %0, %3, %6;"\
"subc.cc.u32 %1, %4, %7;"\
"subc.u32 %2, %5, %8;"\
"}"\
:\
"=r"(c_loc[0]),\
"=r"(c_loc[1]),\
"=r"(c_loc[2])\
:\
"r"(a_loc[0]),\
"r"(a_loc[1]),\
"r"(a_loc[2]),\
"r"(b_loc[0]),\
"r"(b_loc[1]),\
"r"(b_loc[2])\
);\
}

#define sub_glo(c_glo, a_glo, b_glo, tid) {\
asm("{"\
"sub.cc.u32 %0, %3, %6;"\
"subc.cc.u32 %1, %4, %7;"\
"subc.u32 %2, %5, %8;"\
"}"\
:\
"=r"(c_glo[COAL_IDX(0, tid)]),\
"=r"(c_glo[COAL_IDX(1, tid)]),\
"=r"(c_glo[COAL_IDX(2, tid)])\
:\
"r"(a_glo[COAL_IDX(0, tid)]),\
"r"(a_glo[COAL_IDX(1, tid)]),\
"r"(a_glo[COAL_IDX(2, tid)]),\
"r"(b_glo[COAL_IDX(0, tid)]),\
"r"(b_glo[COAL_IDX(1, tid)]),\
"r"(b_glo[COAL_IDX(2, tid)])\
);\
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
// Because of CUDA carry propagation problems (the carry flag is only kept
// for the next assembly instruction), we will have to order the steps in
// the following way:
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
asm("{"\
".reg .u32 %carry;"\
"mul.lo.u32 %0, %3, %5;"\
"add.u32 %carry, 0, 0;"\
"mul.hi.u32 %1, %3, %5;"\
"mad.lo.cc.u32 %1, %3, %6, %1;"\
"addc.u32 %carry, %carry, 0;"\
"mad.lo.cc.u32 %1, %4, %5, %1;"\
"addc.u32 %carry, %carry, 0;"\
"add.u32 %2, %carry, 0;"\
"add.u32 %carry, 0, 0;"\
"mad.hi.cc.u32 %2, %3, %6, %2;"\
"addc.u32 %carry, %carry, 0;"\
"mad.hi.cc.u32 %2, %4, %5, %2;"\
"addc.u32 %carry, %carry, 0;"\
"mad.lo.cc.u32 %2, %4, %6, %2;"\
"}"\
:\
"=r"(c_loc[0]),\
"=r"(c_loc[1]),\
"=r"(c_loc[2])\
:\
"r"(a_loc[0]),\
"r"(a_loc[1]),\
"r"(b_loc[0]),\
"r"(b_loc[2])\
);\
}

#define mul_glo(c_glo, a_glo, b_glo, tid) {\
asm("{"\
".reg .u32 %carry;"\
"mul.lo.u32 %0, %3, %5;"\
"add.u32 %carry, 0, 0;"\
"mul.hi.u32 %1, %3, %5;"\
"mad.lo.cc.u32 %1, %3, %6, %1;"\
"addc.u32 %carry, %carry, 0;"\
"mad.lo.cc.u32 %1, %4, %5, %1;"\
"addc.u32 %carry, %carry, 0;"\
"add.u32 %2, %carry, 0;"\
"add.u32 %carry, 0, 0;"\
"mad.hi.cc.u32 %2, %3, %6, %2;"\
"addc.u32 %carry, %carry, 0;"\
"mad.hi.cc.u32 %2, %4, %5, %2;"\
"addc.u32 %carry, %carry, 0;"\
"mad.lo.cc.u32 %2, %4, %6, %2;"\
"}"\
:\
"=r"(c_glo[COAL_IDX(0, tid)]),\
"=r"(c_glo[COAL_IDX(1, tid)]),\
"=r"(c_glo[COAL_IDX(2, tid)])\
:\
"r"(a_glo[COAL_IDX(0, tid)]),\
"r"(a_glo[COAL_IDX(1, tid)]),\
"r"(b_glo[COAL_IDX(0, tid)]),\
"r"(b_glo[COAL_IDX(2, tid)])\
);\
}
