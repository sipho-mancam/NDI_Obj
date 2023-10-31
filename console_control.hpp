#pragma once
#include <windows.h>
#include <stdio.h>

void setColor(int c);
void gotoxy(int x, int y);
void clrscr();
void box(int y, int x, int w, int h);
void runningAnimation(int x, int y);