#include "bignum_types.h"

#define add_loc(c_loc, a_loc, b_loc) {\
asm("{"\
"add.cc.u32 %0, %9, %18;"\
"addc.cc.u32 %1, %10, %19;"\
"addc.cc.u32 %2, %11, %20;"\
"addc.cc.u32 %3, %12, %21;"\
"addc.cc.u32 %4, %13, %22;"\
"addc.cc.u32 %5, %14, %23;"\
"addc.cc.u32 %6, %15, %24;"\
"addc.cc.u32 %7, %16, %25;"\
"addc.u32 %8, %17, %26;"\
"}"\
:\
"=r"(c_loc[0]),\
"=r"(c_loc[1]),\
"=r"(c_loc[2]),\
"=r"(c_loc[3]),\
"=r"(c_loc[4]),\
"=r"(c_loc[5]),\
"=r"(c_loc[6]),\
"=r"(c_loc[7]),\
"=r"(c_loc[8])\
:\
"r"(a_loc[0]),\
"r"(a_loc[1]),\
"r"(a_loc[2]),\
"r"(a_loc[3]),\
"r"(a_loc[4]),\
"r"(a_loc[5]),\
"r"(a_loc[6]),\
"r"(a_loc[7]),\
"r"(a_loc[8]),\
"r"(b_loc[0]),\
"r"(b_loc[1]),\
"r"(b_loc[2]),\
"r"(b_loc[3]),\
"r"(b_loc[4]),\
"r"(b_loc[5]),\
"r"(b_loc[6]),\
"r"(b_loc[7]),\
"r"(b_loc[8])\
);\
}

#define add_glo(c_glo, a_glo, b_glo, tid) {\
asm("{"\
"add.cc.u32 %0, %9, %18;"\
"addc.cc.u32 %1, %10, %19;"\
"addc.cc.u32 %2, %11, %20;"\
"addc.cc.u32 %3, %12, %21;"\
"addc.cc.u32 %4, %13, %22;"\
"addc.cc.u32 %5, %14, %23;"\
"addc.cc.u32 %6, %15, %24;"\
"addc.cc.u32 %7, %16, %25;"\
"addc.u32 %8, %17, %26;"\
"}"\
:\
"=r"(c_glo[COAL_IDX(0, tid)]),\
"=r"(c_glo[COAL_IDX(1, tid)]),\
"=r"(c_glo[COAL_IDX(2, tid)]),\
"=r"(c_glo[COAL_IDX(3, tid)]),\
"=r"(c_glo[COAL_IDX(4, tid)]),\
"=r"(c_glo[COAL_IDX(5, tid)]),\
"=r"(c_glo[COAL_IDX(6, tid)]),\
"=r"(c_glo[COAL_IDX(7, tid)]),\
"=r"(c_glo[COAL_IDX(8, tid)])\
:\
"r"(a_glo[COAL_IDX(0, tid)]),\
"r"(a_glo[COAL_IDX(1, tid)]),\
"r"(a_glo[COAL_IDX(2, tid)]),\
"r"(a_glo[COAL_IDX(3, tid)]),\
"r"(a_glo[COAL_IDX(4, tid)]),\
"r"(a_glo[COAL_IDX(5, tid)]),\
"r"(a_glo[COAL_IDX(6, tid)]),\
"r"(a_glo[COAL_IDX(7, tid)]),\
"r"(a_glo[COAL_IDX(8, tid)]),\
"r"(b_glo[COAL_IDX(0, tid)]),\
"r"(b_glo[COAL_IDX(1, tid)]),\
"r"(b_glo[COAL_IDX(2, tid)]),\
"r"(b_glo[COAL_IDX(3, tid)]),\
"r"(b_glo[COAL_IDX(4, tid)]),\
"r"(b_glo[COAL_IDX(5, tid)]),\
"r"(b_glo[COAL_IDX(6, tid)]),\
"r"(b_glo[COAL_IDX(7, tid)]),\
"r"(b_glo[COAL_IDX(8, tid)])\
);\
}

#define sub_loc(c_loc, a_loc, b_loc) {\
asm("{"\
"sub.cc.u32 %0, %9, %18;"\
"subc.cc.u32 %1, %10, %19;"\
"subc.cc.u32 %2, %11, %20;"\
"subc.cc.u32 %3, %12, %21;"\
"subc.cc.u32 %4, %13, %22;"\
"subc.cc.u32 %5, %14, %23;"\
"subc.cc.u32 %6, %15, %24;"\
"subc.cc.u32 %7, %16, %25;"\
"subc.u32 %8, %17, %26;"\
"}"\
:\
"=r"(c_loc[0]),\
"=r"(c_loc[1]),\
"=r"(c_loc[2]),\
"=r"(c_loc[3]),\
"=r"(c_loc[4]),\
"=r"(c_loc[5]),\
"=r"(c_loc[6]),\
"=r"(c_loc[7]),\
"=r"(c_loc[8])\
:\
"r"(a_loc[0]),\
"r"(a_loc[1]),\
"r"(a_loc[2]),\
"r"(a_loc[3]),\
"r"(a_loc[4]),\
"r"(a_loc[5]),\
"r"(a_loc[6]),\
"r"(a_loc[7]),\
"r"(a_loc[8]),\
"r"(b_loc[0]),\
"r"(b_loc[1]),\
"r"(b_loc[2]),\
"r"(b_loc[3]),\
"r"(b_loc[4]),\
"r"(b_loc[5]),\
"r"(b_loc[6]),\
"r"(b_loc[7]),\
"r"(b_loc[8])\
);\
}

#define sub_glo(c_glo, a_glo, b_glo, tid) {\
asm("{"\
"sub.cc.u32 %0, %9, %18;"\
"subc.cc.u32 %1, %10, %19;"\
"subc.cc.u32 %2, %11, %20;"\
"subc.cc.u32 %3, %12, %21;"\
"subc.cc.u32 %4, %13, %22;"\
"subc.cc.u32 %5, %14, %23;"\
"subc.cc.u32 %6, %15, %24;"\
"subc.cc.u32 %7, %16, %25;"\
"subc.u32 %8, %17, %26;"\
"}"\
:\
"=r"(c_glo[COAL_IDX(0, tid)]),\
"=r"(c_glo[COAL_IDX(1, tid)]),\
"=r"(c_glo[COAL_IDX(2, tid)]),\
"=r"(c_glo[COAL_IDX(3, tid)]),\
"=r"(c_glo[COAL_IDX(4, tid)]),\
"=r"(c_glo[COAL_IDX(5, tid)]),\
"=r"(c_glo[COAL_IDX(6, tid)]),\
"=r"(c_glo[COAL_IDX(7, tid)]),\
"=r"(c_glo[COAL_IDX(8, tid)])\
:\
"r"(a_glo[COAL_IDX(0, tid)]),\
"r"(a_glo[COAL_IDX(1, tid)]),\
"r"(a_glo[COAL_IDX(2, tid)]),\
"r"(a_glo[COAL_IDX(3, tid)]),\
"r"(a_glo[COAL_IDX(4, tid)]),\
"r"(a_glo[COAL_IDX(5, tid)]),\
"r"(a_glo[COAL_IDX(6, tid)]),\
"r"(a_glo[COAL_IDX(7, tid)]),\
"r"(a_glo[COAL_IDX(8, tid)]),\
"r"(b_glo[COAL_IDX(0, tid)]),\
"r"(b_glo[COAL_IDX(1, tid)]),\
"r"(b_glo[COAL_IDX(2, tid)]),\
"r"(b_glo[COAL_IDX(3, tid)]),\
"r"(b_glo[COAL_IDX(4, tid)]),\
"r"(b_glo[COAL_IDX(5, tid)]),\
"r"(b_glo[COAL_IDX(6, tid)]),\
"r"(b_glo[COAL_IDX(7, tid)]),\
"r"(b_glo[COAL_IDX(8, tid)])\
);\
}

#define mul_loc(c_loc, a_loc, b_loc) {\
asm("{"\
"reg.u32 carry;"\
"mul.lo %0, %9, %14;"\
"add.u32 carry, 0, 0;"\
"mul.hi.u32 %1, %9, %14;"\
"mad.lo.cc.u32 %1, %9, %15, %1;"\
"addc.u32 carry, carry, 0;"\
"mad.lo.cc.u32 %1, %10, %14, %1;"\
"addc.u32 carry, carry, 0;"\
"add.u32 %2, carry, 0;"\
"add.u32 carry, 0, 0;"\
"mad.hi.cc.u32 %2, %9, %15, %2;"\
"addc.u32 carry, carry, 0;"\
"mad.hi.cc.u32 %2, %10, %14, %2;"\
"addc.u32 carry, carry, 0;"\
"mad.lo.cc.u32 %2, %9, %16, %2;"\
"addc.u32 carry, carry, 0;"\
"mad.lo.cc.u32 %2, %10, %15, %2;"\
"addc.u32 carry, carry, 0;"\
"mad.lo.cc.u32 %2, %11, %14, %2;"\
"addc.u32 carry, carry, 0;"\
"add.u32 %3, carry, 0;"\
"add.u32 carry, 0, 0;"\
"mad.hi.cc.u32 %3, %9, %16, %3;"\
"addc.u32 carry, carry, 0;"\
"mad.hi.cc.u32 %3, %10, %15, %3;"\
"addc.u32 carry, carry, 0;"\
"mad.hi.cc.u32 %3, %11, %14, %3;"\
"addc.u32 carry, carry, 0;"\
"mad.lo.cc.u32 %3, %9, %17, %3;"\
"addc.u32 carry, carry, 0;"\
"mad.lo.cc.u32 %3, %10, %16, %3;"\
"addc.u32 carry, carry, 0;"\
"mad.lo.cc.u32 %3, %11, %15, %3;"\
"addc.u32 carry, carry, 0;"\
"mad.lo.cc.u32 %3, %12, %14, %3;"\
"addc.u32 carry, carry, 0;"\
"add.u32 %4, carry, 0;"\
"add.u32 carry, 0, 0;"\
"mad.hi.cc.u32 %4, %9, %17, %4;"\
"addc.u32 carry, carry, 0;"\
"mad.hi.cc.u32 %4, %10, %16, %4;"\
"addc.u32 carry, carry, 0;"\
"mad.hi.cc.u32 %4, %11, %15, %4;"\
"addc.u32 carry, carry, 0;"\
"mad.hi.cc.u32 %4, %12, %14, %4;"\
"addc.u32 carry, carry, 0;"\
"mad.lo.cc.u32 %4, %9, %18, %4;"\
"addc.u32 carry, carry, 0;"\
"mad.lo.cc.u32 %4, %10, %17, %4;"\
"addc.u32 carry, carry, 0;"\
"mad.lo.cc.u32 %4, %11, %16, %4;"\
"addc.u32 carry, carry, 0;"\
"mad.lo.cc.u32 %4, %12, %15, %4;"\
"addc.u32 carry, carry, 0;"\
"mad.lo.cc.u32 %4, %13, %14, %4;"\
"addc.u32 carry, carry, 0;"\
"add.u32 %5, carry, 0;"\
"add.u32 carry, 0, 0;"\
"mad.hi.cc.u32 %5, %9, %18, %5;"\
"addc.u32 carry, carry, 0;"\
"mad.hi.cc.u32 %5, %10, %17, %5;"\
"addc.u32 carry, carry, 0;"\
"mad.hi.cc.u32 %5, %11, %16, %5;"\
"addc.u32 carry, carry, 0;"\
"mad.hi.cc.u32 %5, %12, %15, %5;"\
"addc.u32 carry, carry, 0;"\
"mad.hi.cc.u32 %5, %13, %14, %5;"\
"addc.u32 carry, carry, 0;"\
"mad.lo.cc.u32 %5, %10, %18, %5;"\
"addc.u32 carry, carry, 0;"\
"mad.lo.cc.u32 %5, %11, %17, %5;"\
"addc.u32 carry, carry, 0;"\
"mad.lo.cc.u32 %5, %12, %16, %5;"\
"addc.u32 carry, carry, 0;"\
"mad.lo.cc.u32 %5, %13, %15, %5;"\
"addc.u32 carry, carry, 0;"\
"add.u32 %6, carry, 0;"\
"add.u32 carry, 0, 0;"\
"mad.hi.cc.u32 %6, %10, %18, %6;"\
"addc.u32 carry, carry, 0;"\
"mad.hi.cc.u32 %6, %11, %17, %6;"\
"addc.u32 carry, carry, 0;"\
"mad.hi.cc.u32 %6, %12, %16, %6;"\
"addc.u32 carry, carry, 0;"\
"mad.hi.cc.u32 %6, %13, %15, %6;"\
"addc.u32 carry, carry, 0;"\
"mad.lo.cc.u32 %6, %11, %18, %6;"\
"addc.u32 carry, carry, 0;"\
"mad.lo.cc.u32 %6, %12, %17, %6;"\
"addc.u32 carry, carry, 0;"\
"mad.lo.cc.u32 %6, %13, %16, %6;"\
"addc.u32 carry, carry, 0;"\
"add.u32 %7, carry, 0;"\
"add.u32 carry, 0, 0;"\
"mad.hi.cc.u32 %7, %11, %18, %7;"\
"addc.u32 carry, carry, 0;"\
"mad.hi.cc.u32 %7, %12, %17, %7;"\
"addc.u32 carry, carry, 0;"\
"mad.hi.cc.u32 %7, %13, %16, %7;"\
"addc.u32 carry, carry, 0;"\
"mad.lo.cc.u32 %7, %12, %18, %7;"\
"addc.u32 carry, carry, 0;"\
"mad.lo.cc.u32 %7, %13, %17, %7;"\
"addc.u32 carry, carry, 0;"\
"add.u32 %8, carry, 0;"\
"add.u32 carry, 0, 0;"\
"mad.hi.cc.u32 %8, %12, %18, %8;"\
"addc.u32 carry, carry, 0;"\
"mad.hi.cc.u32 %8, %13, %17, %8;"\
"addc.u32 carry, carry, 0;"\
"mad.lo.cc.u32 %8, %13, %18, %8;"\
"}"\
:\
"=r"(c_loc[0]),\
"=r"(c_loc[1]),\
"=r"(c_loc[2]),\
"=r"(c_loc[3]),\
"=r"(c_loc[4]),\
"=r"(c_loc[5]),\
"=r"(c_loc[6]),\
"=r"(c_loc[7]),\
"=r"(c_loc[8])\
:\
"r"(a_loc[0]),\
"r"(a_loc[1]),\
"r"(a_loc[2]),\
"r"(a_loc[3]),\
"r"(a_loc[4]),\
"r"(b_loc[0]),\
"r"(b_loc[1]),\
"r"(b_loc[2]),\
"r"(b_loc[3]),\
"r"(b_loc[8])\
);\
}

#define mul_glo(c_glo, a_glo, b_glo, tid) {\
asm("{"\
"reg.u32 carry;"\
"mul.lo %0, %9, %14;"\
"add.u32 carry, 0, 0;"\
"mul.hi.u32 %1, %9, %14;"\
"mad.lo.cc.u32 %1, %9, %15, %1;"\
"addc.u32 carry, carry, 0;"\
"mad.lo.cc.u32 %1, %10, %14, %1;"\
"addc.u32 carry, carry, 0;"\
"add.u32 %2, carry, 0;"\
"add.u32 carry, 0, 0;"\
"mad.hi.cc.u32 %2, %9, %15, %2;"\
"addc.u32 carry, carry, 0;"\
"mad.hi.cc.u32 %2, %10, %14, %2;"\
"addc.u32 carry, carry, 0;"\
"mad.lo.cc.u32 %2, %9, %16, %2;"\
"addc.u32 carry, carry, 0;"\
"mad.lo.cc.u32 %2, %10, %15, %2;"\
"addc.u32 carry, carry, 0;"\
"mad.lo.cc.u32 %2, %11, %14, %2;"\
"addc.u32 carry, carry, 0;"\
"add.u32 %3, carry, 0;"\
"add.u32 carry, 0, 0;"\
"mad.hi.cc.u32 %3, %9, %16, %3;"\
"addc.u32 carry, carry, 0;"\
"mad.hi.cc.u32 %3, %10, %15, %3;"\
"addc.u32 carry, carry, 0;"\
"mad.hi.cc.u32 %3, %11, %14, %3;"\
"addc.u32 carry, carry, 0;"\
"mad.lo.cc.u32 %3, %9, %17, %3;"\
"addc.u32 carry, carry, 0;"\
"mad.lo.cc.u32 %3, %10, %16, %3;"\
"addc.u32 carry, carry, 0;"\
"mad.lo.cc.u32 %3, %11, %15, %3;"\
"addc.u32 carry, carry, 0;"\
"mad.lo.cc.u32 %3, %12, %14, %3;"\
"addc.u32 carry, carry, 0;"\
"add.u32 %4, carry, 0;"\
"add.u32 carry, 0, 0;"\
"mad.hi.cc.u32 %4, %9, %17, %4;"\
"addc.u32 carry, carry, 0;"\
"mad.hi.cc.u32 %4, %10, %16, %4;"\
"addc.u32 carry, carry, 0;"\
"mad.hi.cc.u32 %4, %11, %15, %4;"\
"addc.u32 carry, carry, 0;"\
"mad.hi.cc.u32 %4, %12, %14, %4;"\
"addc.u32 carry, carry, 0;"\
"mad.lo.cc.u32 %4, %9, %18, %4;"\
"addc.u32 carry, carry, 0;"\
"mad.lo.cc.u32 %4, %10, %17, %4;"\
"addc.u32 carry, carry, 0;"\
"mad.lo.cc.u32 %4, %11, %16, %4;"\
"addc.u32 carry, carry, 0;"\
"mad.lo.cc.u32 %4, %12, %15, %4;"\
"addc.u32 carry, carry, 0;"\
"mad.lo.cc.u32 %4, %13, %14, %4;"\
"addc.u32 carry, carry, 0;"\
"add.u32 %5, carry, 0;"\
"add.u32 carry, 0, 0;"\
"mad.hi.cc.u32 %5, %9, %18, %5;"\
"addc.u32 carry, carry, 0;"\
"mad.hi.cc.u32 %5, %10, %17, %5;"\
"addc.u32 carry, carry, 0;"\
"mad.hi.cc.u32 %5, %11, %16, %5;"\
"addc.u32 carry, carry, 0;"\
"mad.hi.cc.u32 %5, %12, %15, %5;"\
"addc.u32 carry, carry, 0;"\
"mad.hi.cc.u32 %5, %13, %14, %5;"\
"addc.u32 carry, carry, 0;"\
"mad.lo.cc.u32 %5, %10, %18, %5;"\
"addc.u32 carry, carry, 0;"\
"mad.lo.cc.u32 %5, %11, %17, %5;"\
"addc.u32 carry, carry, 0;"\
"mad.lo.cc.u32 %5, %12, %16, %5;"\
"addc.u32 carry, carry, 0;"\
"mad.lo.cc.u32 %5, %13, %15, %5;"\
"addc.u32 carry, carry, 0;"\
"add.u32 %6, carry, 0;"\
"add.u32 carry, 0, 0;"\
"mad.hi.cc.u32 %6, %10, %18, %6;"\
"addc.u32 carry, carry, 0;"\
"mad.hi.cc.u32 %6, %11, %17, %6;"\
"addc.u32 carry, carry, 0;"\
"mad.hi.cc.u32 %6, %12, %16, %6;"\
"addc.u32 carry, carry, 0;"\
"mad.hi.cc.u32 %6, %13, %15, %6;"\
"addc.u32 carry, carry, 0;"\
"mad.lo.cc.u32 %6, %11, %18, %6;"\
"addc.u32 carry, carry, 0;"\
"mad.lo.cc.u32 %6, %12, %17, %6;"\
"addc.u32 carry, carry, 0;"\
"mad.lo.cc.u32 %6, %13, %16, %6;"\
"addc.u32 carry, carry, 0;"\
"add.u32 %7, carry, 0;"\
"add.u32 carry, 0, 0;"\
"mad.hi.cc.u32 %7, %11, %18, %7;"\
"addc.u32 carry, carry, 0;"\
"mad.hi.cc.u32 %7, %12, %17, %7;"\
"addc.u32 carry, carry, 0;"\
"mad.hi.cc.u32 %7, %13, %16, %7;"\
"addc.u32 carry, carry, 0;"\
"mad.lo.cc.u32 %7, %12, %18, %7;"\
"addc.u32 carry, carry, 0;"\
"mad.lo.cc.u32 %7, %13, %17, %7;"\
"addc.u32 carry, carry, 0;"\
"add.u32 %8, carry, 0;"\
"add.u32 carry, 0, 0;"\
"mad.hi.cc.u32 %8, %12, %18, %8;"\
"addc.u32 carry, carry, 0;"\
"mad.hi.cc.u32 %8, %13, %17, %8;"\
"addc.u32 carry, carry, 0;"\
"mad.lo.cc.u32 %8, %13, %18, %8;"\
"}"\
:\
"=r"(c_glo[COAL_IDX(0, tid)]),\
"=r"(c_glo[COAL_IDX(1, tid)]),\
"=r"(c_glo[COAL_IDX(2, tid)]),\
"=r"(c_glo[COAL_IDX(3, tid)]),\
"=r"(c_glo[COAL_IDX(4, tid)]),\
"=r"(c_glo[COAL_IDX(5, tid)]),\
"=r"(c_glo[COAL_IDX(6, tid)]),\
"=r"(c_glo[COAL_IDX(7, tid)]),\
"=r"(c_glo[COAL_IDX(8, tid)])\
:\
"r"(a_glo[COAL_IDX(0, tid)]),\
"r"(a_glo[COAL_IDX(1, tid)]),\
"r"(a_glo[COAL_IDX(2, tid)]),\
"r"(a_glo[COAL_IDX(3, tid)]),\
"r"(a_glo[COAL_IDX(4, tid)]),\
"r"(b_glo[COAL_IDX(0, tid)]),\
"r"(b_glo[COAL_IDX(1, tid)]),\
"r"(b_glo[COAL_IDX(2, tid)]),\
"r"(b_glo[COAL_IDX(3, tid)]),\
"r"(b_glo[COAL_IDX(8, tid)])\
);\
}

