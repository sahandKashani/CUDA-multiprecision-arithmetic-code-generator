#ifndef _PTX_INLINE_
#define _PTX_INLINE_


#define __add(r,a,b){\
    asm volatile("add.u32 %0, %1, %2;" : "=r" (r) : "r" (a), "r" (b):"memory");\
}

#define __addc(r,a,b){\
    asm volatile("addc.u32 %0, %1, %2;" : "=r" (r):  "r" (a),  "r" (b):"memory");\
}

#define __add_cc(r,a,b){\
    asm volatile("add.cc.u32 %0, %1, %2;" : "=r" (r) : "r" (a), "r" (b):"memory");\
}

#define __addc_cc(r,a,b){\
    asm volatile("addc.cc.u32 %0, %1, %2;" : "=r" (r) : "r" (a), "r" (b):"memory");\
}

#define __sub(r,a,b){\
    asm volatile("sub.u32 %0, %1, %2;" : "=r" (r) : "r" (a), "r" (b):"memory");\
}

#define __subc(r,a,b){\
    asm volatile("subc.u32 %0, %1, %2;" : "=r" (r) : "r" (a), "r" (b):"memory");\
}

#define __sub_cc(r,a,b){\
    asm volatile("sub.cc.u32 %0, %1, %2;" : "=r" (r) : "r" (a), "r" (b):"memory");\
}

#define __subc_cc(r,a,b){\
    asm volatile("subc.cc.u32 %0, %1, %2;" : "=r" (r) : "r" (a), "r" (b):"memory");\
}

#define __subc2(a,b){\
    asm volatile("subc.u32 %0, %0, %1;" : "+r" (a) : "r" (b):"memory");\
}

#define __subc_cc2(a,b){\
    asm volatile("subc.cc.u32 %0, %0, %1;" : "+r" (a) : "r" (b):"memory");\
}

#define __sub2(a,b){\
    asm volatile("sub.u32 %0, %0, %1;" : "+r" (a) : "r" (b):"memory");\
}

#define __sub_cc2(a,b){\
    asm volatile("sub.cc.u32 %0, %0, %1;" : "+r" (a) : "r" (b):"memory");\
}

#define __addc2(a,b){\
    asm volatile("addc.u32 %0, %0, %1;" : "+r" (a) : "r" (b):"memory");\
}

#define __addc_cc2(a,b){\
    asm volatile("addc.cc.u32 %0, %0, %1;" : "+r" (a) : "r" (b):"memory");\
}

#define __add2(a,b){\
    asm volatile("add.u32 %0, %0, %1;" : "+r" (a) : "r" (b):"memory");\
}

#define __add_cc2(a,b){\
    asm volatile("add.cc.u32 %0, %0, %1;" : "+r" (a) : "r" (b):"memory");\
}

#define __mull(r,a,b){\
    asm volatile("mul.lo.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b):"memory");\
}

#define __mulh(r,a,b){\
    asm volatile("mul.hi.u32 %0, %1, %2;" : "=r"(r) : "r"(a) "r"(b):"memory");\
}

#define  __muladdh_cc(r,a,b,c){\
    asm volatile("mad.hi.cc.u32 %0, %1, %2, %3;" : "=r"(r)  "+r"(a):"r"(b), "r"(c):"memory");\
}

#endif
