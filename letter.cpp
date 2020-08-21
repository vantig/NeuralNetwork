#include "Header.h"
#include <iostream>
#include <time.h>
#include <immintrin.h>
#include <algorithm>
#include <SFML/Graphics.hpp>
#include <SFML/Window.hpp>
#include <fstream>
using namespace std;
using namespace sf;
void letter()
{
    struct s1 {
        double img[64][64];
        int rAnswer;
    };
    RenderWindow window(VideoMode(320, 320), "letter");
    Image img;
    img.create(64, 64);
    Texture t;
    Sprite s;
    t.loadFromImage(img);
    s.setTexture(t);
    Mouse m;
    s.setPosition(0, 0);

    for (int i = 0; i < 64; i++)
    {
        for (int j = 0; j < 64; j++) 
        {
            img.setPixel(i, j, Color::White);
        }
    }
    int radius = 4;

    while (window.isOpen())
    {
        Event event;
        while (window.pollEvent(event))
        {
            if (event.type == Event::Closed)
            {
                window.close();
            }

        }

        if (Mouse::isButtonPressed(Mouse::Left))
        {
            if ((Mouse::getPosition().x < window.getPosition().x + 320) and (Mouse::getPosition().x > window.getPosition().x))
            {
                if ((Mouse::getPosition().y < window.getPosition().y + 320) and (Mouse::getPosition().y > window.getPosition().y))
                {
                    int x = Mouse::getPosition().x;
                    int y = Mouse::getPosition().y;
                    for (int i = x - radius; i < x + radius; i++)
                    {
                        for (int j = y - radius; j < y + radius; j++)
                        {
                            img.setPixel((i - window.getPosition().x) / 5, (j - window.getPosition().y - 25) / 5, Color::Black);
                        }
                    }
                }


            }
        }
        t.loadFromImage(img);
        s.setTexture(t);
        s.setScale(5.0, 5.0);
        window.clear();
        window.draw(s);
        window.display();
    }
    //img.saveToFile("img.jpg");
    ofstream fout;
    fout.open("letter.txt");

    s1 obj1;
    obj1.rAnswer = 0;
    for (int i = 0; i < 64; i++) {
        for (int j = 0; j < 64; j++) {
            obj1.img[i][j] = (1.0-img.getPixel(j, i).r / 255.0);
            cout << obj1.img[i][j] << " ";
            fout << obj1.img[i][j] << " ";
        }
        cout << endl;
        fout << endl;
    }
    fout.close();
}

