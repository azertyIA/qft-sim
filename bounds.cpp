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

#define COLOR true

void write_frame(Catrix h, GLuint tex);
void cat_pulse(Catrix A, const float mx, const float my, const float2 k);

int main(int argc, char *argv[]) {
  const float dt = std::stof(argv[4]);
  const int fps = 144;
  const int cols = std::stoi(argv[1]);
  const int rows = std::stoi(argv[2]);
  const int St = std::stoi(argv[3]);

  Catrix A = cat_alloc_host(rows, cols);
  Catrix U = cat_adapt(A);
  Catrix V = cat_adapt(A);
  Catrix dnext = cat_adapt(A);

  SDL_Init(SDL_INIT_VIDEO);
  SDL_Window *window = SDL_CreateWindow("Wave Sim", SDL_WINDOWPOS_CENTERED,
                                        SDL_WINDOWPOS_CENTERED, A.cols, A.rows,
                                        SDL_WINDOW_OPENGL);
  SDL_GLContext glctx = SDL_GL_CreateContext(window);
  SDL_GL_SetSwapInterval(1);

  GLuint tex;
  glGenTextures(1, &tex);
  glBindTexture(GL_TEXTURE_2D, tex);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, cols, rows, 0, GL_RGB,
               GL_UNSIGNED_BYTE, nullptr);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

  // initialization
  cat_fill(U, 0);
  cat_fill(V, 0);
  cat_to_host(A, V);

  for (int row = 0; row < rows; row++) {
    for (int col = 0; col < cols; col++) {
      if (col * 9 < 4 * cols || col * 9 > 5 * cols)
        continue;
      CAT_AT(A, row, col) = make_float2(1.0f, 0.0f); // real part 1, imag 0
    }
  }
  cat_to_device(V, A);

  bool running = true;
  Uint32 last = SDL_GetTicks();
  SDL_Event e;

  for (int i = 0; running; i++) {
    cat_step(dnext, U, V, dt);
    float2 *tmp = U.data;
    U.data = dnext.data;
    dnext.data = tmp;

    while (SDL_PollEvent(&e)) {
      if (e.type == SDL_QUIT)
        running = false;

      if (e.type == SDL_MOUSEBUTTONDOWN) {
        int mx, my;
        SDL_GetMouseState(&mx, &my);

        cat_to_host(A, U);
        cat_pulse(A, mx, my, make_float2(2, 0));
        cat_to_device(U, A);
      }
    }

    if (i % St != 0)
      continue;

    cat_to_host(A, U);
    write_frame(A, tex);

    auto v = CAT_AT(A, 10, 10);
    float maxamp = 0;
    for (size_t i = 0; i < A.rows * A.cols; i++) {
      float2 z = A.data[i];
      float mag = sqrtf(z.x * z.x + z.y * z.y);
      if (mag > maxamp)
        maxamp = mag;
    }
    printf("step %d max |psi| = %g\n", i, maxamp);

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

    Uint32 now = SDL_GetTicks();
    Uint32 frameTime = now - last;
    last = SDL_GetTicks();
  }

  cat_free_host(A);
  cat_free_device(U);
  cat_free_device(V);
  cat_free_device(dnext);

  SDL_GL_DeleteContext(glctx);
  SDL_DestroyWindow(window);
  SDL_Quit();
  return 0;
}

void cat_rgb(unsigned char *img, const Catrix u, const float scale) {
  for (size_t row = 0; row < u.rows; row++) {
    for (size_t col = 0; col < u.cols; col++) {
      float2 v = CAT_AT(u, row, col);
      const float hue = atan2f(v.y, v.x) / M_PI / 2 + 0.5f;
      const float mag = fminf(sqrtf(v.x * v.x + v.y * v.y) * scale, 1);

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
      // if (COLOR) {
      img[idx + 0] = (unsigned char)(255 * rgb.x * mag);
      img[idx + 1] = (unsigned char)(255 * rgb.y * mag);
      img[idx + 2] = (unsigned char)(255 * rgb.z * mag);
      // return;
      // }
      // img[idx + 0] = (unsigned char)(255 * mag);
      // img[idx + 1] = (unsigned char)(255 * mag);
      // img[idx + 2] = (unsigned char)(255 * mag);
    }
  }
}

void cat_pulse(Catrix A, const float mx, const float my, const float2 k) {
  for (int x = 0; x < A.cols; x++) {
    for (int y = 0; y < A.rows; y++) {
      float dx = x - mx;
      float dy = y - my;
      float sigma = 8.0f;
      float amp = expf(-(dx * dx + dy * dy) / (2 * sigma * sigma));
      float phi = k.x * dx + k.y * dy;
      CAT_AT(A, A.rows - y, x) += make_float2(amp * cosf(phi), amp * sinf(phi));
    }
  }
}

void write_frame(Catrix h, GLuint tex) {
  float maxval = 0.5f;

  unsigned char img[h.rows * h.cols * 3];
  cat_rgb(img, h, maxval);
  glBindTexture(GL_TEXTURE_2D, tex);
  glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, h.cols, h.rows, GL_RGB,
                  GL_UNSIGNED_BYTE, img);
}
