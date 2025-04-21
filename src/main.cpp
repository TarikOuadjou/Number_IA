#include <app/screen.h>
#include <core/neural.h>
#include <iostream>
#include <string>
#include <sstream>

int main(int argc, char *argv[])
{
    Neural network({784, 128, 64, 10}, "relu");
    network.Import("mnist_model_neural.txt", "relu");
    Screen screen(network);
    while (true)
    {
        if (screen.flag_taille == 0)
        {
            screen.input();
            screen.show_large();
            screen.draw_point_souris_large();
        }
        else
        {
            screen.show_mini();
            screen.input();
            screen.draw_point_souris_mini();
        }
    }
    return 0;
}
