#ifndef PTX_INLINE_OPERATIONS_H
#define PTX_INLINE_OPERATIONS_H

#include "bignum_types.h"

#define ptx_add(c, a, b, tid)                                                  \
{                                                                              \
    asm("add.cc.u32 %0, %1, %2;"                                               \
        : "=r"(c[COAL_IDX(0, tid)])                                            \
        : "r" (a[COAL_IDX(0, tid)]),                                           \
          "r" (b[COAL_IDX(0, tid)]));                                          \
                                                                               \
    for (uint32_t i = 1; i < BIGNUM_NUMBER_OF_WORDS - 1; i++)                  \
    {                                                                          \
        asm("addc.cc.u32 %0, %1, %2;"                                          \
            : "=r"(c[COAL_IDX(i, tid)])                                        \
            : "r" (a[COAL_IDX(i, tid)]),                                       \
              "r" (b[COAL_IDX(i, tid)]));                                      \
    }                                                                          \
                                                                               \
    asm("addc.u32 %0, %1, %2;"                                                 \
        : "=r"(c[COAL_IDX(BIGNUM_NUMBER_OF_WORDS - 1, tid)])                   \
        : "r" (a[COAL_IDX(BIGNUM_NUMBER_OF_WORDS - 1, tid)]),                  \
          "r" (b[COAL_IDX(BIGNUM_NUMBER_OF_WORDS - 1, tid)]));                 \
}

#define ptx_sub(c, a, b, tid)                                                  \
{                                                                              \
    asm("sub.cc.u32 %0, %1, %2;"                                               \
        : "=r"(c[COAL_IDX(0, tid)])                                            \
        : "r" (a[COAL_IDX(0, tid)]),                                           \
          "r" (b[COAL_IDX(0, tid)]));                                          \
                                                                               \
    for (uint32_t i = 1; i < BIGNUM_NUMBER_OF_WORDS - 1; i++)                  \
    {                                                                          \
        asm("subc.cc.u32 %0, %1, %2;"                                          \
            : "=r"(c[COAL_IDX(i, tid)])                                        \
            : "r" (a[COAL_IDX(i, tid)]),                                       \
              "r" (b[COAL_IDX(i, tid)]));                                      \
    }                                                                          \
                                                                               \
    asm("subc.u32 %0, %1, %2;"                                                 \
        : "=r"(c[COAL_IDX(BIGNUM_NUMBER_OF_WORDS - 1, tid)])                   \
        : "r" (a[COAL_IDX(BIGNUM_NUMBER_OF_WORDS - 1, tid)]),                  \
          "r" (b[COAL_IDX(BIGNUM_NUMBER_OF_WORDS - 1, tid)]));                 \
}

#endif
