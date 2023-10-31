#pragma once

#include "console_control.hpp"

void setColor(int c)
{
    HANDLE hConsoleHandle = GetStdHandle(STD_OUTPUT_HANDLE);
    SetConsoleTextAttribute(hConsoleHandle, c);
}

void gotoxy(int x, int y)
{
    COORD pos = { x, y };
    HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
    SetConsoleCursorPosition(hConsole, pos);
}


void clrscr()
{
    HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
    CONSOLE_SCREEN_BUFFER_INFO csbi;
    int rows, columns;
    GetConsoleScreenBufferInfo(hConsole, &csbi);

    columns = csbi.srWindow.Right - csbi.srWindow.Left + 1;
    rows = csbi.srWindow.Bottom - csbi.srWindow.Top + 1;

    int i, j;
    for (i = 0; i < rows; i++)
    {
        for (j = 0; j < columns; j++)
        {
            gotoxy(j, i);
            printf(" ");
        }
    }
    gotoxy(0, 0);
}


void box(int y, int x, int w, int h)
{
    for (size_t width = 0; width <= w; ++width)
    {
        gotoxy(x + 1 + (width - 1), y);
        printf("%c", 205);
        gotoxy(x + 1 + (width - 1), y + h);
        printf("%c", 205);
    }
    for (size_t height = 0; height < h; ++height)
    {
        gotoxy(x, (height - 1) + y + 1);
        printf("%c", 186);
        gotoxy(x + w, (height - 1) + y + 1);
        printf("%c", 186);
    }
    gotoxy(x, y);
    printf("%c", 201);
    gotoxy(x + w, y);
    printf("%c", 187);
    gotoxy(x, y + h);
    printf("%c", 200);
    gotoxy(x + w, y + h);
    printf("%c", 188);

}

void runningAnimation(int x, int y)
{

}