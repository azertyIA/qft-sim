#include "cat.h"

typedef struct {
  const float h_bar, m, q, delta_time;
  const SFieldFn ax, ay, o;
} QuantumGaugeParams;

typedef struct {
  float2 AV_FACTOR, V2_FACTOR, H_FACTOR;
  SField prev, curr, next;
  VField A;
  SField O, A2, VA, AVU, V2U;
  SField H, HU;
  float2 *tmp;
} QuantumGauge;

typedef struct {
  float D2_FACTOR, H_FACTOR;
  SField prev, curr, next;
  VField A, Tr, Td;
  VField TrU, TdU;
  SField O, H;
  float2 *tmp;
} CovariantGauge;

QuantumGauge qg_alloc(SField host);
void qg_dump(SField dump, QuantumGauge gauge);
void qg_load(QuantumGauge gauge, SField host);
void qg_free(QuantumGauge gauge);

void qg_init(QuantumGauge &gauge, const QuantumGaugeParams params, SField host);
void qg_step_second_order(QuantumGauge &gauge);
void qg_step_so_single(QuantumGauge &gauge);
void qg_step_decomp(QuantumGauge &gauge);
void qg_step_wave(QuantumGauge &gauge);

CovariantGauge cg_alloc(SField host);
void cg_dump(SField dump, CovariantGauge gauge);
void cg_load(CovariantGauge gauge, SField host);
void cg_free(CovariantGauge gauge);

void cg_init(CovariantGauge &gauge, const QuantumGaugeParams params,
             SField host);
void cg_step_second_order(CovariantGauge &gauge);
// void cg_step_so_single(CovariantGauge &gauge);
