#!/usr/bin/env python

from collections import deque

from helpers import XY

class CommandRegulator:
    def __init__(self, Pk, Dk):
        self.Pk = Pk
        self.Dk = Dk
        self.previous_commands = deque(maxlen=3)

    def set_coeffs(self, Pk, Dk):
        self.Pk = Pk
        self.Dk = Dk


    def get_coeffs(self):
        return (self.Pk, self.Dk)


    def next_command(self, command : XY, dt_ms : float):
        assert(isinstance(command, XY))
        commands = self.previous_commands
        if len(commands) == 0:
            commands.append(command)
            return command

        prev_command : XY = self.previous_commands[-1]
        commands.append(command)

        Pk, Dk = self.Pk, self.Dk

        # Note: looks like introducing D leads to crazy command output, too big
        return command * Pk #+ (command - prev_command) * (Dk / dt_ms)


def test_CommandRegulator():
    c = CommandRegulator(12, 0.2)
    print(c.nextCommand(XY(1, 2), 100))
    print(c.nextCommand(XY(2, 3), 100))



if __name__ == "__main__":
    test_CommandRegulator()