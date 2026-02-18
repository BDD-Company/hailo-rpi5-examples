#!/usr/bin/env python3

from helpers import *

def main():
    x=0.170
    y=0.428
    w=0.041
    h=0.056
    r = Rect.from_xyxy(x, y, x + w, y + h)
    # assert r.width == w
    # assert r.height == h
    print(r.area())


if __name__ == "__main__":
    main()