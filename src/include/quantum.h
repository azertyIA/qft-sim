#include "cat.h"

typedef struct {
  const float h_bar, m, q, delta_time;
  const SFieldFn ax, ay, o;
} QuantumGaugeParams;

typedef struct {
  float2 AV_FACTOR, V2_FACTOR, H_FACTOR;
  SField prev, curr, next;
  VField A, VU;
  SField O, A2, VA, AVU, V2U;
  SField H, HU;
  float2 *tmp;
} QuantumGauge;

QuantumGauge qg_alloc(SField host);
void qg_dump(SField dump, QuantumGauge gauge);
void qg_load(QuantumGauge gauge, SField host);
void qg_free(QuantumGauge gauge);

void qg_init(QuantumGauge gauge, const QuantumGaugeParams params);
void qg_step_first_order(QuantumGauge gauge);
void qg_step_second_order(QuantumGauge gauge);
