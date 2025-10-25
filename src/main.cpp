#include "cat.h"
#include <GL/gl.h>
#include <SDL2/SDL.h>
#include <SDL2/SDL_events.h>
#include <SDL2/SDL_mouse.h>
#include <SDL2/SDL_stdinc.h>
#include <SDL2/SDL_timer.h>
#include <SDL2/SDL_video.h>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <hip/amd_detail/amd_hip_vector_types.h>
#include <string>

// #define COLOR
// #define READ_V

// physical constants
const float q = -0.02f;
const float dV = 1.0f;

void init_window(Catrix H, SDL_Window *window, SDL_GLContext *glctx,
                 GLuint *tex);
void write_frame(Catrix h, GLuint tex, SDL_Window *window);
void cat_pulse(Catrix A, const float mx, const float my, const float2 k);

Catrix init_perturbation(Catrix h) {
  Catrix V0 = cat_adapt(h);

  for (int row = 0; row < h.rows; row++)
    for (int col = 0; col < h.cols; col++)
      CAT_AT(h, row, col) = q * dV * make_float2((float)col / h.cols, 0);

  cat_to_device(V0, h);
  return V0;
}

void on_click(Catrix h, Catrix T, Catrix U, int x0, int y0, int x, int y) {
  int dy = y - y0;
  int dx = x - x0;
  float n = sqrt(dy * dy + dx * dx);

  cat_to_host(h, U);
  cat_pulse(h, x, y, 2.0f * make_float2(dx, dy) / n);
  cat_to_device(U, h);
  cat_to_device(T, h);
}

int main(int argc, char *argv[]) {
  const int cols = std::stoi(argv[1]);
  const int rows = std::stoi(argv[2]);
  const int St = std::stoi(argv[3]);
  const float dt = std::stof(argv[4]);

  // host matrix for reading / writing
  Catrix H = cat_alloc_host(rows, cols);

  // buffer matrices
  Catrix U0 = cat_adapt(H);
  Catrix U1 = cat_adapt(H);
  Catrix U2 = cat_adapt(H);

  // potiential matrices
  Catrix V0 = init_perturbation(H);
  Catrix V = cat_adapt(H);

  // initialization
  cat_fill(U0, 0);
  cat_fill(U1, 0);
  cat_fill(V, 0);

  // user events
  bool running = true;
  SDL_Event e;
  int x0 = -1, y0 = -1, x1, y1, vx, vy;
  float n;

  // intitalize window
  GLuint *tex;
  SDL_Init(SDL_INIT_VIDEO);
  SDL_Window *window = SDL_CreateWindow("Wave Sim", SDL_WINDOWPOS_CENTERED,
                                        SDL_WINDOWPOS_CENTERED, H.cols, H.rows,
                                        SDL_WINDOW_OPENGL);
  SDL_GLContext glctx = (SDL_GL_CreateContext(window));
  SDL_GL_SetSwapInterval(1);

  glGenTextures(1, tex);
  glBindTexture(GL_TEXTURE_2D, *tex);
  return 0;
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, H.cols, H.rows, 0, GL_RGB,
               GL_UNSIGNED_BYTE, nullptr);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

  for (int i = 0; running; i++) {
    cat_norm(V, U1, -q);
    cat_add(V, V, V0);
    cat_step(U2, U0, U1, V, dt);
    float2 *tmp = U0.data;
    U0.data = U1.data;
    U1.data = U2.data;
    U2.data = tmp;

    if (i % St != 0)
      continue;

    while (SDL_PollEvent(&e)) {
      if (e.type == SDL_QUIT)
        running = false;
      if (e.type == SDL_MOUSEBUTTONDOWN)
        SDL_GetMouseState(&x0, &y0);
      if (e.type != SDL_MOUSEBUTTONUP)
        continue;
      SDL_GetMouseState(&x1, &y1);

      if (x0 == -1 || x0 == x1 || y0 == y1)
        continue;
      on_click(H, U0, U1, x0, y0, x1, y1);
    }

    cat_to_host(H, U1);
    write_frame(H, *tex, window);

#ifdef READ_V
    SDL_GetMouseState(&vx, &vy);
    cat_to_host(H, V0);
    float v0 = CAT_AT(H, vy, vx).x;
    printf("V=%.2f x=%d y=%d", v0, vx, vy);
#endif // READ_V
  }

  // free matrices
  cat_free_host(H);
  cat_free_device(U0);
  cat_free_device(U1);
  cat_free_device(U2);
  cat_free_device(V0);
  cat_free_device(V);

  SDL_GL_DeleteContext(glctx);
  SDL_DestroyWindow(window);
  SDL_Quit();
  return 0;
}

void init_window(Catrix H, SDL_Window *window, SDL_GLContext *glctx,
                 GLuint *tex) {
  SDL_Init(SDL_INIT_VIDEO);
  window = SDL_CreateWindow("Wave Sim", SDL_WINDOWPOS_CENTERED,
                            SDL_WINDOWPOS_CENTERED, H.cols, H.rows,
                            SDL_WINDOW_OPENGL);
  *glctx = SDL_GL_CreateContext(window);
  SDL_GL_SetSwapInterval(1);

  glGenTextures(1, tex);
  glBindTexture(GL_TEXTURE_2D, *tex);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, H.cols, H.rows, 0, GL_RGB,
               GL_UNSIGNED_BYTE, nullptr);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
}

void cat_rgb(unsigned char *img, const Catrix u, const float scale) {
  for (size_t row = 0; row < u.rows; row++) {
    for (size_t col = 0; col < u.cols; col++) {
      float2 v = CAT_AT(u, row, col);
      const float hue = atan2f(v.y, v.x) / M_PI / 2 + 0.5f;
      // const float mag = fminf(sqrtf(v.x * v.x + v.y * v.y) * scale, 1);
      const float mag =
          fmax(0, fminf(sqrtf(v.x * v.x + v.y * v.y) * scale, 1) - 0.05f);

      const float h = hue * 6.0f;
      const float f = h - floorf(h);
      float3 rgb;
      switch ((int)h) {
      case 0:
        rgb = make_float3(1, f, 0);
        break;
      case 1:
        rgb = make_float3(1 - f, 1, 0);
        break;
      case 2:
        rgb = make_float3(0, 1, f);
        break;
      case 3:
        rgb = make_float3(0, 1 - f, 1);
        break;
      case 4:
        rgb = make_float3(f, 0, 1);
        break;
      case 5:
        rgb = make_float3(1, 0, 1 - f);
        break;
      }
      int idx = (row * u.cols + col) * 3;
#ifdef COLOR
      img[idx + 0] = (unsigned char)(255 * rgb.x * mag);
      img[idx + 1] = (unsigned char)(255 * rgb.y * mag);
      img[idx + 2] = (unsigned char)(255 * rgb.z * mag);
#else
      img[idx + 0] = (unsigned char)(255 * mag);
      img[idx + 1] = (unsigned char)(255 * mag);
      img[idx + 2] = (unsigned char)(255 * mag);
#endif // COLOR
    }
  }
}

void cat_pulse(Catrix A, const float mx, const float my, const float2 k) {
  for (int x = 0; x < A.cols; x++) {
    for (int y = 0; y < A.rows; y++) {
      float dx = x - mx;
      float dy = y - my;
      float sigma = 6.0f;
      float amp = 0.4f * expf(-(dx * dx + dy * dy) / (2 * sigma * sigma));
      float phi = k.x * dx + k.y * dy;
      CAT_AT(A, A.rows - y, x) += make_float2(amp * cosf(phi), amp * sinf(phi));
    }
  }
}

void write_frame(Catrix h, GLuint tex, SDL_Window *window) {
  float maxval = 0.5f;

  unsigned char img[h.rows * h.cols * 3];
  cat_rgb(img, h, maxval);
  glBindTexture(GL_TEXTURE_2D, tex);
  glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, h.cols, h.rows, GL_RGB,
                  GL_UNSIGNED_BYTE, img);
  glClear(GL_COLOR_BUFFER_BIT);
  glEnable(GL_TEXTURE_2D);
  glBindTexture(GL_TEXTURE_2D, tex);
  glBegin(GL_QUADS);
  glTexCoord2f(0, 0);
  glVertex2f(-1, -1);
  glTexCoord2f(1, 0);
  glVertex2f(1, -1);
  glTexCoord2f(1, 1);
  glVertex2f(1, 1);
  glTexCoord2f(0, 1);
  glVertex2f(-1, 1);
  glEnd();
  SDL_GL_SwapWindow(window);
}
