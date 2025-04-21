/* Include files*/
#include <SDL2/SDL.h>
#include <core/knn.h>
#include <core/neural.h>
#include <vector>
#include <cmath>
#include <iostream>

class Screen
{
public:
    Screen(Neural network);

    void show_mini();

    void show_large();

    void input();

    void efface();

    ~Screen();

    void draw_point_souris_mini();

    void draw_point_souris_large();

    int flag_taille;

private:
    SDL_Event e;
    SDL_Window *window;
    SDL_Renderer *renderer;
    int mouse_x;
    int mouse_y;
    int flag_appuyer;
    double array[28][28];
    int array_large[560][560];
    int radius_peinture;
    std::vector<MNISTData> dataset;
    Neural network;
};