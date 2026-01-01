#include "quantum.h"
#include <cstdio>

QuantumGauge qg_alloc(SField host) {
  QuantumGauge gauge;

  // potential fields
  gauge.A = {s_adapt(host), s_adapt(host)};
  gauge.O = s_adapt(host);

  // wave field buffers
  gauge.prev = s_adapt(host);
  gauge.curr = s_adapt(host);
  gauge.next = s_adapt(host);

  // intermediate calculations (static)
  gauge.A2 = s_adapt(host);
  gauge.VA = s_adapt(host);

  // intermediate calculations (dynamic)
  gauge.AVU = s_adapt(host);
  gauge.V2U = s_adapt(host);

  // hamiltonians (static, dynamic)
  gauge.H = s_adapt(host);
  gauge.HU = s_adapt(host);
  return gauge;
}

void qg_dump(SField dump, QuantumGauge gauge) { s_to_host(dump, gauge.curr); }
void qg_load(QuantumGauge gauge, SField host) {
  s_to_device(gauge.curr, host);
  s_to_device(gauge.prev, host);
};

void qg_free(QuantumGauge gauge) {
  s_free_device(gauge.A.x);
  s_free_device(gauge.A.y);
  s_free_device(gauge.O);

  s_free_device(gauge.prev);
  s_free_device(gauge.curr);
  s_free_device(gauge.next);

  s_free_device(gauge.A2);
  s_free_device(gauge.VA);

  s_free_device(gauge.AVU);
  s_free_device(gauge.V2U);

  s_free_device(gauge.H);
  s_free_device(gauge.HU);
}

void qg_init(QuantumGauge &gauge, const QuantumGaugeParams params,
             SField host) {
  gauge.AV_FACTOR = make_float2(0, -params.h_bar / params.m);
  gauge.V2_FACTOR =
      make_float2(-0.5f * params.h_bar * params.h_bar / params.m, 0);
  gauge.H_FACTOR = make_float2(0, -2 * params.delta_time / params.h_bar);

  s_host_fill(host, params.ax);
  s_to_device(gauge.A.x, host);
  s_host_fill(host, params.ay);
  s_to_device(gauge.A.y, host);
  s_host_fill(host, params.o);
  s_to_device(gauge.O, host);

  s_scale(gauge.A.x, gauge.A.x, params.q);
  s_scale(gauge.A.y, gauge.A.y, params.q);
  s_scale(gauge.O, gauge.O, params.q);

  s_fill(gauge.prev, 0);
  s_fill(gauge.curr, 0);
  s_fill(gauge.next, 0);

  f_dot(gauge.A2, gauge.A, gauge.A);
  s_scale(gauge.A2, gauge.A2, 0.5f / params.m);

  f_div(gauge.VA, gauge.A);
  s_scale(gauge.VA, gauge.VA, make_float2(0, 0.5f * params.h_bar / params.m));

  // precompute H = [(ihV.(qA) + (qA)2)/2m + O]
  s_add(gauge.H, gauge.O, gauge.A2);
  s_add(gauge.H, gauge.H, gauge.VA);

  // skip A.VU scaling
  s_scale(gauge.A.x, gauge.A.x, gauge.AV_FACTOR);
  s_scale(gauge.A.y, gauge.A.y, gauge.AV_FACTOR);
}

void qg_step_second_order(QuantumGauge &gauge) {
  // HU += [(ihV.(qA) + (qA)2)/2m + O]U
  s_mult(gauge.HU, gauge.H, gauge.curr);

  // HU += (-ihqA/m).VU
  f_dot_grad(gauge.AVU, gauge.A, gauge.curr);
  s_add(gauge.HU, gauge.HU, gauge.AVU);

  // HU += -h2V2U/2m
  s_lap(gauge.V2U, gauge.curr);
  s_scale_add(gauge.HU, gauge.HU, gauge.V2U, gauge.V2_FACTOR);

  // U+ = U- - 2iHdt/h U
  s_scale_add(gauge.next, gauge.prev, gauge.HU, gauge.H_FACTOR);

  gauge.tmp = gauge.prev.data;
  gauge.prev.data = gauge.curr.data;
  gauge.curr.data = gauge.next.data;
  gauge.next.data = gauge.tmp;
}

void qg_step_so_single(QuantumGauge &gauge) {
  q_step_so(gauge.next, gauge.curr, gauge.prev, gauge.H, gauge.A,
            gauge.V2_FACTOR, gauge.H_FACTOR);

  gauge.tmp = gauge.prev.data;
  gauge.prev.data = gauge.curr.data;
  gauge.curr.data = gauge.next.data;
  gauge.next.data = gauge.tmp;
}

void qg_step_decomp(QuantumGauge &gauge) {
  // HU += [(ihV.(qA) + (qA)2)/2m + O]U
  s_mult(gauge.VA, gauge.H, gauge.curr);

  // HU += (-ihqA/m).VU
  f_dot_grad(gauge.AVU, gauge.A, gauge.curr);
  s_add(gauge.HU, gauge.VA, gauge.AVU);

  // HU += -h2V2U/2m
  s_lap(gauge.V2U, gauge.curr);
  s_scale(gauge.V2U, gauge.V2U, gauge.V2_FACTOR);
  s_add(gauge.HU, gauge.HU, gauge.V2U);

  // U+ = U- - 2iHdt/h U
  s_scale_add(gauge.next, gauge.prev, gauge.HU, gauge.H_FACTOR);

  gauge.tmp = gauge.prev.data;
  gauge.prev.data = gauge.curr.data;
  gauge.curr.data = gauge.next.data;
  gauge.next.data = gauge.tmp;
}

void qg_step_wave(QuantumGauge &gauge) {
  // HU += -h2V2U/2m
  s_lap(gauge.V2U, gauge.curr);
  s_scale(gauge.HU, gauge.V2U, gauge.V2_FACTOR);
  s_scale(gauge.HU, gauge.HU, gauge.H_FACTOR);
  s_add(gauge.next, gauge.prev, gauge.HU);

  gauge.tmp = gauge.prev.data;
  gauge.prev.data = gauge.curr.data;
  gauge.curr.data = gauge.next.data;
  gauge.next.data = gauge.tmp;
}
