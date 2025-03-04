# import pygame
class Game:
    STAY = 0
    LEFT = 1
    RIGHT = 2

    def __init__(self):
        # start at the top and move down
        self.level = [
            (0, 0, 0),
            (0, 0, 0),
            (0, 1, 0),
            (0, 0, 0),
            (0, 0, 0),
            (1, 1, 0),
            (0, 0, 0),
            (0, 0, 0),
            (0, 1, 1),
            (0, 0, 0),
            (0, 0, 0),
        ]

        self.position = (0, 1)

    def step(self, move):
        # if move == self.LEFT
        pass

    def print(self):
        for i in range(len(self.level)):
            row = ["-" if j==1 else " " for j in self.level[i]]
            if self.position[0] == i:
                row[self.position[1]] = "X"
            print(f"|{"".join(row)}|")



g = Game()
g.print()