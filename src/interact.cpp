#include "mat.h"
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
#include <string>
#include <vector>

void write_frame(Matrix h, GLuint tex) {
  float maxval = 0.1f;

  static std::vector<unsigned char> img;
  img.resize(h.rows * h.cols * 3);

  for (size_t j = 0; j < h.rows; ++j) {
    for (size_t i = 0; i < h.cols; ++i) {
      float v = MAT_AT(h, j, i) / maxval; // [-1,1]
      v = fminf(fmaxf(v, -1.0f), 1.0f);

      float r = (v > 0) ? v : 0.0f;
      float b = (v < 0) ? -v : 0.0f;
      float g = 0; // neutral background

      size_t idx = 3 * (j * h.cols + i);
      img[idx + 0] = (unsigned char)(255 * g);
      img[idx + 1] = (unsigned char)(255 * r);
      img[idx + 2] = (unsigned char)(255 * b);
    }
  }

  glBindTexture(GL_TEXTURE_2D, tex);
  glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, h.cols, h.rows, GL_RGB,
                  GL_UNSIGNED_BYTE, img.data());
}

int main(int argc, char *argv[]) {
  const float c2 = std::stof(argv[4]);
  const int fps = 144;
  const int Nx = std::stoi(argv[1]);
  const int Ny = std::stoi(argv[2]);
  const int St = std::stoi(argv[3]);

  Matrix A = mat_alloc_host(Ny, Nx);

  SDL_Init(SDL_INIT_VIDEO);
  SDL_Window *window = SDL_CreateWindow("Wave Sim", SDL_WINDOWPOS_CENTERED,
                                        SDL_WINDOWPOS_CENTERED, A.cols, A.rows,
                                        SDL_WINDOW_OPENGL);
  SDL_GLContext glctx = SDL_GL_CreateContext(window);
  SDL_GL_SetSwapInterval(1);

  GLuint tex;
  glGenTextures(1, &tex);
  glBindTexture(GL_TEXTURE_2D, tex);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, A.cols, A.rows, 0, GL_RGB,
               GL_UNSIGNED_BYTE, nullptr);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

  Matrix dA = mat_adapt(A);
  Matrix dB = mat_adapt(A);
  Matrix dC = mat_adapt(A);

  // initialization
  mat_fill_host(A, 0);
  mat_copy_to_device(dA, A);
  mat_copy_to_device(dB, A);

  bool running = true;
  Uint32 last = SDL_GetTicks();
  SDL_Event e;

  for (int i = 0; running; i++) {
    mat_wave_device(dC, dB, dA, c2);
    float *tmp = dA.data;
    dA.data = dB.data;
    dB.data = dC.data;
    dC.data = tmp;

    mat_copy_to_host(A, dB);
    float ct = sin(0.01 * (float)i);
    for (int col = 0; col < A.cols; col++) {
      MAT_AT(A, 2, col) += ct;
    }

    while (SDL_PollEvent(&e)) {
      if (e.type == SDL_QUIT)
        running = false;

      if (e.type == SDL_MOUSEBUTTONDOWN) {
        int mx, my;
        SDL_GetMouseState(&mx, &my);
        MAT_AT(A, A.rows - my, mx) += 1;
      }
    }

    mat_copy_to_device(dB, A);

    if (i % St != 0)
      continue;
    mat_copy_to_host(A, dA);

    write_frame(A, tex);
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
    // Uint32 target = 1000 / fps;
    // if (frameTime < target)
    //   SDL_Delay(target - frameTime);
    last = SDL_GetTicks();
  }

  mat_free_host(A);
  mat_free_device(dA);
  mat_free_device(dB);
  mat_free_device(dC);

  SDL_GL_DeleteContext(glctx);
  SDL_DestroyWindow(window);
  SDL_Quit();
  return 0;
}
