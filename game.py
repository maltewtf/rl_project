# import pygame
class Game:
    STAY = 0
    LEFT = -1
    RIGHT = 1

    def __init__(self):
        # start at the top and move down
        self.width = 3
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

        self.position = (0, 1) # y-value, x-value

    def step(self, move):
        move = max(min(int(move), 1), -1)
        self.position = (self.position[0] + 1, 
                        max(min(self.position[1] + move, self.width-1), 0))

    def print(self):
        for i in range(len(self.level)):
            row = ["-" if j==1 else " " for j in self.level[i]]
            if self.position[0] == i:
                row[self.position[1]] = "X"
            print(f"|{''.join(row)}|")

    def run(self):
        while self.position[0] < len(self.level) - 1:
            self.print()
            self.step(input("move: "))
            if self.level[self.position[0]][self.position[1]] == 1:
                print(self.position[0])
                return self.position[0]
        print(self.position[0])
        return self.position[0]
            
g = Game()
g.run()
