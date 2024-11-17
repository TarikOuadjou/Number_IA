#include <app/screen.h>
#include <core/neural.h>
#include <iostream> 
#include <string>

int main(int argc, char* argv[]) {

    Screen screen;
    while(true)
    {
        if(screen.flag_taille==0)
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
    /*std::vector<int> vec = {2,3,3,2};
    Neural neural(vec);
    neural.afficher();*/
    return 0;
}

