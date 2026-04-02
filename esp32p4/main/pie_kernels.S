.section .text
    .align  4

    .global pie_dot_s8
    .type   pie_dot_s8, @function
pie_dot_s8:
    mv      a4, a3
    li      a3, 1
    j       pie_matmul_s8_xacc
    .size   pie_dot_s8, . - pie_dot_s8

    .global pie_matmul_s8_xacc
    .type   pie_matmul_s8_xacc, @function

pie_matmul_s8_xacc:
    addi    sp, sp, -28
    sw      s0,  0(sp)
    sw      s1,  4(sp)
    sw      s2,  8(sp)
    sw      s3, 12(sp)
    sw      s4, 16(sp)
    sw      s5, 20(sp)
    sw      s6, 24(sp)

    mv      s0, a0
    mv      s1, a1
    mv      s2, a2
    mv      s3, a3
    mv      s4, a4

    beqz    s3, .Ldone

    srli    s5, s4, 5
    slli    t0, s5, 5
    sub     s6, s4, t0
    snez    s6, s6

.Lrow_loop:
    esp.zero.xacc

    mv      a1, s0
    mv      a2, s1

    esp.vld.128.ip  q0, a1, 16
    esp.vld.128.ip  q1, a2, 16

    mv      t0, s5
    beqz    t0, .Ltail_check

.Lbulk_loop:
    esp.vld.128.ip  q2, a1, 16
    esp.vmulas.s8.xacc.ld.ip  q3, a2, 16, q0, q1

    esp.vld.128.ip  q0, a1, 16
    esp.vmulas.s8.xacc.ld.ip  q1, a2, 16, q2, q3

    addi    t0, t0, -1
    bnez    t0, .Lbulk_loop

.Ltail_check:
    beqz    s6, .Lextract

    esp.vmulas.s8.xacc  q0, q1

.Lextract:
    li      a3, 0
    esp.srs.s.xacc  t3, a3

    sw      t3, 0(s2)
    addi    s2, s2, 4

    add     s0, s0, s4
    addi    s3, s3, -1
    bnez    s3, .Lrow_loop

.Ldone:
    lw      s0,  0(sp)
    lw      s1,  4(sp)
    lw      s2,  8(sp)
    lw      s3, 12(sp)
    lw      s4, 16(sp)
    lw      s5, 20(sp)
    lw      s6, 24(sp)
    addi    sp, sp, 28
    ret

    .size pie_matmul_s8_xacc, . - pie_matmul_s8_xacc