#include "screen.h"
#include <fstream>
#include <sstream>

void calcule_gris(int array_large[560][560], double array_mini[28][28])
{
    double somme = 0;
    for (int i = 0; i < 28; i++)
    {
        for (int j = 0; j < 28; j++)
        {
            for (int k = 0; k < 20; k++)
            {
                for (int l = 0; l < 20; l++)
                {
                    somme += array_large[20 * i + k][20 * j + l];
                }
            }
            array_mini[i][j] = somme / (20.0 * 20.0);
            somme = 0;
        }
    }
}

const std::vector<double> &calcule_vecteur(double array_mini[28][28])
{
    static std::vector<double> result;
    result.clear();
    for (int i = 0; i < 28; ++i)
    {
        for (int j = 0; j < 28; ++j)
        {
            result.push_back(static_cast<double>(array_mini[j][i]));
        }
    }
    return result;
}

void print_array(double array[28][28])
{
    for (int i = 0; i < 28; ++i)
    {
        for (int j = 0; j < 28; ++j)
        {
            std::cout << array[j][i] << " ";
        }
        std::cout << std::endl;
    }
}

void draw_circle(int center_x, int center_y, int radius, int array_large[560][560])
{
    int x = 0;
    int y = radius;
    int d = 3 - 2 * radius;

    auto set_array_points = [&](int x, int y)
    {
        for (int i = center_x - x; i <= center_x + x; i++)
        {
            if (i >= 0 && i < 560 && center_y + y >= 0 && center_y + y < 560)
                array_large[i][center_y + y] = 1;
            if (i >= 0 && i < 560 && center_y - y >= 0 && center_y - y < 560)
                array_large[i][center_y - y] = 1;
        }
        for (int i = center_x - y; i <= center_x + y; i++)
        {
            if (i >= 0 && i < 560 && center_y + x >= 0 && center_y + x < 560)
                array_large[i][center_y + x] = 1;
            if (i >= 0 && i < 560 && center_y - x >= 0 && center_y - x < 560)
                array_large[i][center_y - x] = 1;
        }
    };

    while (y >= x)
    {
        set_array_points(x, y);
        x++;

        if (d > 0)
        {
            y--;
            d = d + 4 * (x - y) + 10;
        }
        else
        {
            d = d + 4 * x + 6;
        }

        set_array_points(x, y);
    }
}

std::vector<MNISTData> readMNISTCSV(const std::string &filePath)
{
    std::vector<MNISTData> dataset;
    std::ifstream file(filePath);

    if (!file.is_open())
    {
        throw std::runtime_error("Impossible d'ouvrir le fichier : " + filePath);
    }

    std::string line;

    while (std::getline(file, line))
    {
        std::stringstream ss(line);
        std::string value;
        MNISTData data;

        std::getline(ss, value, ',');
        data.label = std::stoi(value);

        while (std::getline(ss, value, ','))
        {
            int pixel = std::stoi(value);
            // Normalisation du pixel
            data.pixels.push_back(static_cast<float>(pixel) / 255.0f);
        }

        dataset.push_back(data);
    }
    file.close();
    return dataset;
}

Screen::Screen(Neural network) : network(network)
{
    if (SDL_Init(SDL_INIT_VIDEO) != 0)
    {
        std::cerr << "Erreur SDL_Init: " << SDL_GetError() << std::endl;
        exit(1);
    }

    if (SDL_CreateWindowAndRenderer(840, 560, 0, &window, &renderer) != 0)
    {
        std::cerr << "Erreur SDL_CreateWindowAndRenderer: " << SDL_GetError() << std::endl;
        SDL_Quit();
        exit(1);
    }
    for (int i = 0; i < 28; i++)
    {
        for (int j = 0; j < 28; j++)
        {
            array[i][j] = 0;
        }
    }
    for (int i = 0; i < 560; i++)
    {
        for (int j = 0; j < 560; j++)
        {
            array_large[i][j] = 0;
        }
    }
    flag_appuyer = 0;
    mouse_x = 0;
    mouse_y = 0;
    radius_peinture = 10;
    flag_taille = 0;
    const std::string filePath = "mnist_train.csv";
    dataset = readMNISTCSV(filePath);
}

void Screen::show_mini()
{
    SDL_Rect rect;
    rect.x = 560;
    rect.y = 0;
    rect.w = 280;
    rect.h = 840;
    SDL_SetRenderDrawColor(renderer, 0, 14, 56, SDL_ALPHA_OPAQUE);
    SDL_RenderClear(renderer);
    SDL_SetRenderDrawColor(renderer, 0, 14, 30, 255); // Couleur du rectangle (rouge)
    SDL_RenderFillRect(renderer, &rect);
    for (int i = 0; i < 28; i++)
    {
        for (int j = 0; j < 28; j++)
        {
            SDL_SetRenderDrawColor(renderer, (int)(array[i][j] * 255), (int)(array[i][j] * 255), (int)(array[i][j] * 255), 255);
            SDL_RenderDrawPoint(renderer, 20 * i, 20 * j);
            SDL_Rect rect_ij;
            rect_ij.x = 20 * i;
            rect_ij.y = 20 * j;
            rect_ij.w = 20;
            rect_ij.h = 20;
            SDL_RenderFillRect(renderer, &rect_ij);
        }
    }
    SDL_SetRenderDrawColor(renderer, 0, 0, 0, SDL_ALPHA_OPAQUE);
    SDL_RenderPresent(renderer);
}

void Screen::show_large()
{
    SDL_Rect rect;
    rect.x = 560;
    rect.y = 0;
    rect.w = 280;
    rect.h = 840;
    SDL_SetRenderDrawColor(renderer, 0, 14, 56, SDL_ALPHA_OPAQUE);
    SDL_RenderClear(renderer);
    SDL_SetRenderDrawColor(renderer, 0, 14, 30, 255); // Couleur du rectangle (bleu foncé)
    SDL_RenderFillRect(renderer, &rect);
    for (int i = 0; i < 560; i++)
    {
        for (int j = 0; j < 560; j++)
        {
            SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);
            if (array_large[i][j] == 1)
            {
                SDL_RenderDrawPoint(renderer, i, j);
            }
        }
    }
    SDL_SetRenderDrawColor(renderer, 0, 0, 0, SDL_ALPHA_OPAQUE);
    SDL_RenderPresent(renderer);
}

void Screen::draw_point_souris_mini()
{
    if (flag_appuyer == 1)
    {
        array[(int)mouse_x / 20][(int)mouse_y / 20] = 1;
    }
}

void Screen::draw_point_souris_large()
{
    if (flag_appuyer == 1)
    {
        draw_circle(mouse_x, mouse_y, radius_peinture, array_large);
    }
}

void Screen::efface()
{
    for (int i = 0; i < 28; i++)
    {
        for (int j = 0; j < 28; j++)
        {
            array[i][j] = 0;
        }
    }
    for (int i = 0; i < 560; i++)
    {
        for (int j = 0; j < 560; j++)
        {
            array_large[i][j] = 0;
        }
    }
}

void Screen::input()
{
    while (SDL_PollEvent(&e))
    {
        if (e.type == SDL_QUIT)
        {
            SDL_Quit();
            exit(0);
        }
        else if (e.type == SDL_MOUSEMOTION)
        {
            mouse_x = e.motion.x;
            mouse_y = e.motion.y;
        }
        else if (e.type == SDL_MOUSEBUTTONDOWN)
        {
            if (e.button.button == SDL_BUTTON_LEFT)
            {
                flag_appuyer = 1;
            }
        }
        else if (e.type == SDL_MOUSEBUTTONUP)
        {
            if (e.button.button == SDL_BUTTON_LEFT)
            {
                flag_appuyer = 0;
            }
        }
        else if (e.type == SDL_KEYDOWN)
        {
            if (e.key.keysym.sym == SDLK_e)
            {
                Screen::efface();
            }
            if (e.key.keysym.sym == SDLK_k)
            {
                calcule_gris(array_large, array);
                const std::vector<double> &input = calcule_vecteur(array);
                int k = 5;
                int predictedLabel = predictLabel(dataset, input, k);
                std::cout << "la reponse est : " << predictedLabel << std::endl;
            }

            if (e.key.keysym.sym == SDLK_n)
            {
                calcule_gris(array_large, array);
                const std::vector<double> &input = calcule_vecteur(array);
                int predictedLabel = network.Predict(input);
                std::cout << "la reponse est : " << predictedLabel << std::endl;
            }

            if (e.key.keysym.sym == SDLK_a)
            {
                calcule_gris(array_large, array);
                print_array(array);
                flag_taille = 1 - flag_taille;
            }
        }
        else if (e.type == SDL_MOUSEWHEEL)
        {
            radius_peinture = e.wheel.y + radius_peinture;
        }
    }
}

Screen::~Screen()
{
    // Destruction du renderer et de la fenêtre
    if (renderer)
    {
        SDL_DestroyRenderer(renderer);
    }
    if (window)
    {
        SDL_DestroyWindow(window);
    }
    SDL_Quit();
}