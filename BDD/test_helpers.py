#!/usr/bin/env python3

from helpers import *


def test_xy_format():
    """XY format specifiers are applied to x and y individually."""
    v = XY(1.23456, 0.5)

    assert f'{v}'       == 'XY(1.235, 0.500)'
    assert f'{v:.2f}'   == 'XY(1.23, 0.50)'
    assert f'{v:.3f}'   == 'XY(1.235, 0.500)'
    assert f'{v: .3}'   == 'XY( 1.23,  0.5)'
    assert f'{v:>8.2f}' == 'XY(    1.23,     0.50)'
    assert str(v)       == 'XY(1.235, 0.500)'
    assert repr(v)      == 'XY(x=1.23456, y=0.5)'


def main():
    test_xy_format()

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