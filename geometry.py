import math

class Vec2:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __mul__(self, n):
        return Vec2(self.x * n, self.y * n)

    def __div__(self, n):
        return Vec2(self.x / n, self.y / n)

    def __truediv__(self, n):
        return Vec2(self.x / n, self.y / n)

    def __add__(self, v):
        return Vec2(self.x + v.x, self.y + v.y)

    def __sub__(self, v):
        return Vec2(self.x - v.x, self.y - v.y)

    # def __div__(self, v):
    #     return Vec2(self.x // v.x, self.y // v.y)

    def get(self):
        return (self.x, self.y)

    def get_length(self):
        return math.sqrt(self.x**2 + self.y**2)

    def set(self, v):
        self.x, self.y = v.x, v.y

class Rect:
    def __init__(self, pos, size):
        self.pos = pos
        self.size = size

    def get_rect(self):
        return self.pos.get() + self.size.get()

class Collider(Rect):
    def __init__(self, pos, size):
        Rect.__init__(self, pos, size)
    
    def collide_rect(self, pos, size, vel = Vec2(0, 0), frametime=1):
        xreset, yreset = 0,0

        x, y = pos.get()
        w, h = size.get()
        xv, yv = (vel * frametime).get() 

        # x += xv

        # if self.pos.x + self.size.x >= x + w >= self.pos.x and self.pos.y + self.size.y + h - 1 >= y + h >= self.pos.y + 1:
        #     xreset = (x + w) - self.pos.x

        # elif self.pos.x + self.size.x >= x >= self.pos.x and self.pos.y + self.size.y + h - 1 >= y + h >= self.pos.y + 1:
        #     xreset = x - self.pos.x - self.size.x
        
        # x -= xv
        # y += yv

        # if self.pos.y + self.size.y >= y + h >= self.pos.y and self.pos.x + self.size.x + w - 1 >= x + w >= self.pos.x + 1:
        #     yreset = (y + h) - self.pos.y - yv

        # elif self.pos.y + self.size.y >= y >= self.pos.y and self.pos.x + self.size.x + w - 1 >= x + w >= self.pos.x + 1:
        #     yreset = y - self.pos.y - self.size.y - yv


        if xv > 0:
            # approach from left
            if x + xv + w > self.pos.x > x and self.pos.y + self.size.y + h >= y + h >= self.pos.y:
                xreset = x + w - self.pos.x
            
        elif xv < 0:
            # approach from right
            if x + xv < self.pos.x + self.size.x < x + w and self.pos.y + self.size.y + h>= y + h >= self.pos.y:
                xreset = x - self.pos.x - self.size.x   
        
        if yv > 0:
            # approach from top
            if y + yv + h > self.pos.y > y and self.pos.x + self.size.x + w >= x + w >= self.pos.x:
                yreset = y + h - self.pos.y
            
        elif yv < 0:
            # approach from bottom
            if y + yv < self.pos.y + self.size.y < y + h and self.pos.x + self.size.x + w >= x + w >= self.pos.x:
                yreset = y - self.pos.y - self.size.y

        x -= xv
        y += yv

        return xreset, yreset

    def collide_legacy(self, pos, size, vel = Vec2(0, 0), frametime=1):

        x, y, w, h, xv, yv = pos.get() + size.get() + vel.get()

        xreset, yreset = 0,0
        x += xv
        if self.pos.x + self.size.x >= x + w >= self.pos.x and self.pos.y + self.size.y + h - 20 >= y + h >= self.pos.y + 20:
            xreset = (x + w) - self.pos.x
        elif self.pos.x + self.size.x >= x >= self.pos.x and self.pos.y + self.size.y + h - 20 >= y + h >= self.pos.y + 20:
            xreset = x - self.pos.x - self.size.x
        x -= xv
        y += yv
        if self.pos.y + self.size.y >= y + h >= self.pos.y and self.pos.x + self.size.x + w - 20 >= x + w >= self.pos.x + 20:
            yreset = (y + h) - self.pos.y
        elif self.pos.y + self.size.y >= y >= self.pos.y and self.pos.x + self.size.x + w - 20 >= x + w >= self.pos.x + 20:
            yreset = y - self.pos.y - self.size.y

        return xreset, yreset