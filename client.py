import pygame
import queue
import numpy as np
import time
import camInput
from threading import Thread
from geometry import *
from keybinds import Keybinds

class Log:
    def __init__(self):
        self.logged_data = []
        self.log_level = 0
        self.space = 0
        self.info = 1
        self.warn = 2
        self.error = 3
        self.logs = ("      ", "[...] ", "[!!!] ", "[ERROR]")

    def log(self, data, priority):
        if priority >= self.log_level:
            try: print(self.logs[priority % len(self.logs)] + str(data))
            except: print("[NO_LOG_LVL] " + str(data))
            self.logged_data.append((priority, str(data)))

    def set_log_level(self, level):
        self.log_level = level

    def get_log(self):
        return self.logged_data

class Cam:
    def __init__(self, win, pos=Vec2(0, 0), size=Vec2(1920, 1080)):
        self.win = win
        self.pos = pos
        self.size = size
        self.smoothing = 0
        self.scale = 64

        self.xmax = 0
        self.ymax = 0
        self.xmin = 0
    
    def set_borders(self, objects):
        for i in objects:
            if i.pos.x + i.size.x - self.size.x/64 > self.xmax:
                self.xmax = i.pos.x + i.size.x - self.size.x/64
        for i in objects:
            if i.pos.x < self.xmin:
                self.xmin = i.pos.x
        for i in objects:
            if i.pos.y > self.ymax:
                self.ymax = i.pos.y + i.size.y
    
    def set_pos(self, pos):
        self.pos.x, self.pos.y = max(min(pos.x, self.xmax * 64), self.xmin * 64), min(pos.y, self.ymax * 64)
        # self.pos.x, self.pos.y = pos.x, pos.y

class AnimatedTexture:
    def __init__(self, textures, speed = 10):
        self.textures = textures
        self.speed = speed

class Block(Collider):
    def __init__(self, pos, size, texture = None, mode = 0, dmg = 0):
        super().__init__(pos, size)
        self.texture = texture
        self.mode = mode
        self.dmg = dmg

        if texture == None:
            self.texture = default

    def draw(self, cam):
        relx, rely = cam.pos.get()
        if self.mode == 1:
            return
        elif self.mode == 3:
            cam.win.blit(self.texture, (self.x + relx, self.y + rely))
        else:
            for i in range(int(self.size.x/64)):
                for a in range(int(self.size.y/64)):
                    pos = (self.pos + Vec2(i * 64, a * 64)) - cam.pos
                    # print(pos.get())
                    cam.win.blit(self.texture, pos.get())

class Level:
    def __init__(self, name="level2"):
        self.name = name
        self.objects = []
        self.background = None
        self.spawn = Vec2(0, 0)

    def load(self):
        import LevelReader
        # except: log.log("Can't find level loading tool >>> please add python file to folder!", log.warn); return
        blocks, entities = LevelReader.read("levels/" + self.name)
        self.decode_level(blocks)

    def get_grids(self):
        pass

    def get_objects(self):
        return self.objects

    def get_entities(self):
        pass

    def drawBackground(self, screen, cam):
        if self.background: #replace
            relx = -(cam.pos.x - round(cam.pos.x/64 - 1)*64)
            rely = -(cam.pos.y - round(cam.pos.y/64 - 1)*64)
            for i in range(round(screen[0]/32+4)):
                for a in range(round(screen[1]/32+3)):
                    cam.win.blit(self.background, (0 + i*32 + relx, 0 + a*32 + rely))
        else:
            cam.win.fill((64, 64, 64))

    def decode_level(self, blocks):
        print(blocks)
        obj_list = []
        entity_list = []
        for i in blocks:
            # if the element is a list its data for a block
            if type(i) == list:
                try:
                    try:
                        try:
                            img = pygame.transform.scale(pygame.image.load("sprites/" + i[4]), (64, 64))
                            obj_list.append(Block(Vec2(i[0],i[1]) * 64, Vec2(i[2],i[3]) * 64, img, i[5], i[6]))
                        except Exception as e:
                            img = pygame.transform.scale(pygame.image.load("sprites/" + i[4]), (64, 64))
                            obj_list.append(Block(Vec2(i[0],i[1]) * 64, Vec2(i[2],i[3]) * 64, img))
                            log.log("Mode Missing! >>> " + str(e), log.warn)
                    except Exception as e:
                        obj_list.append(Block(Vec2(i[0],i[1]) * 64, Vec2(i[2],i[3]) * 64, pygame.transform.scale(default, (64, 64))))
                        log.log("Texture Missing! >>> " + str(e), log.warn)
                except Exception as e:
                    log.log("Loading Error! >>> " + str(e), log.error)
                    log.log("Skipping!", log.warn)
            # if its not a list (a string) its metadata
            else:
                metadata = i.split("=")
                try:
                    # check for different kinds of metadata like backgroundtexture and goal/spawn positions
                    if metadata[0] == "background":
                        try:
                            self.background = pygame.transform.scale(pygame.image.load("sprites/" + metadata[1]), (32, 32))
                        except:
                            log.log("Background missing!", log.warn)

                    elif metadata[0] == "goal":
                        x, y = [float(i)*64 for i in metadata[1].split(",")]
                        # self.goal.set_pos(x, y)

                    elif metadata[0] == "spawn":
                        x, y = [float(i)*64 for i in metadata[1].split(",")]
                        self.spawn = Vec2(x, y)
                except:
                    log.log("invalid Metadata: " + str(i), 1)

        # Xmax, Xmin, Ymax = self.borders(obj_list)
        log.log("loading completed! " + str(len(obj_list)) + " Objects Loaded", log.info)

        self.objects = obj_list
        # print(obj_list)
        # self.obj_list, self.entity_list, self.Xmax, self.Xmin, self.Ymax = obj_list, entity_list, Xmax, Xmin, Ymax

class Grid:
    # basic data structure that stores smaller parts of a level
    def __init__(self):
        pass

class Player(Collider):
    def __init__(self, pos = Vec2(0, 0), size = Vec2(48, 88)):
        super().__init__(pos, size)
        self.vel = Vec2(0, 0)
        self.alive = True
        self.crouched = False
        self.on_ground = False
        self.jump_cooldown = 0
        self.state = self.sub_state = 0
        self.animation_counter = 0
        self.facing_backwards = False
        self.load()
        # self.texture = pygame.transform.scale(pygame.image.load("player_sprites/Player_Idle1.png"), (48, 88))
        # self.small = pygame.transform.scale(pygame.image.load("player_sprites/Player_crouched.png"), (48, 60))

    def load(self):
        player_idle1 = pygame.transform.scale(pygame.image.load("player_sprites/Player_Idle1.png"), (48, 88))
        player_idle2 = pygame.transform.scale(pygame.image.load("player_sprites/Player_Idle2.png"), (48, 88))
        player_idle3 = pygame.transform.scale(pygame.image.load("player_sprites/Player_Idle3.png"), (48, 88))

        player_run1 = pygame.transform.scale(pygame.image.load("player_sprites/Player_Idle1.png"), (48, 88))
        player_run2 = pygame.transform.scale(pygame.image.load("player_sprites/Player_Idle2.png"), (48, 88))
        player_run3 = pygame.transform.scale(pygame.image.load("player_sprites/Player_Idle3.png"), (48, 88))

        player_jump1 = pygame.transform.scale(pygame.image.load("player_sprites/Player_jump1.png"), (48, 88))
        player_jump2 = pygame.transform.scale(pygame.image.load("player_sprites/Player_jump2.png"), (48, 88))
        player_jump3 = pygame.transform.scale(pygame.image.load("player_sprites/Player_jump3.png"), (48, 88))

        player_falling = pygame.transform.scale(pygame.image.load("player_sprites/Player_falling.png"), (48, 88))

        player_falling1 = pygame.transform.scale(pygame.image.load("player_sprites/Player_falling1.png"), (48, 88))
        player_falling2 = pygame.transform.scale(pygame.image.load("player_sprites/Player_falling2.png"), (48, 88))
        player_falling3 = pygame.transform.scale(pygame.image.load("player_sprites/Player_falling3.png"), (48, 88))

        player_crouched = pygame.transform.scale(pygame.image.load("player_sprites/Player_crouched.png"), (48, 60))

        self.animations = [[player_idle1, player_idle2, player_idle3],
                           [player_run1, player_run2, player_run3],
                           [player_jump1, player_jump2, player_jump3],
                           [player_falling1, player_falling2, player_falling3],
                           [player_crouched]]

        self.state_speeds = [400, 50, 100, 100, 50]

    # calculates the state (e.g. idle, crouched, running, etc.) of annimation the playermodel is in
    def calc_state(self):
        # sets direction the palyer is facing
        if self.vel.x < 0:
            self.facing_backwards = False
        elif self.vel.x > 0:
            self.facing_backwards = True

        old = self.state
        if self.size.y <= 80: # is crouching
            self.state = 4
        elif self.vel.y == 0 and self.vel.y == 0 and self.state != 0: # is idle
            self.state = 0
        elif self.vel.x != 0 and self.vel.y == 0 and self.state != 1: # is running
            self.state = 1
        elif self.vel.y < 0 and self.state != 2: # is jumping
            self.state = 2
        elif self.vel.y > 0 and self.state != 3: # is falling
            self.state = 3

        if old != self.state:
            self.animation_counter = 0

    # calculates the sub-state (frame) of annimation the playermodel is in
    def calc_sub_state(self, frametime, time = 50):
        if self.animation_counter <= 0:
            self.sub_state += 1
            self.animation_counter = time
        else:
            self.animation_counter -= frametime

        self.sub_state %= len(self.animations[self.state])

    def draw(self, cam):
        tempIMG = self.animations[self.state][self.sub_state]
        tempIMG = pygame.transform.flip(tempIMG, not self.facing_backwards, False)
        cam.win.blit(tempIMG, (self.pos.x - cam.pos.x, self.pos.y - cam.pos.y))

    def move(self, ft, level):
        Potxv, Potyv = self.input(ft, self.vel.x)
        xreset, yreset = 0, 0 

        # moves animation
        self.calc_state(); 
        self.calc_sub_state(ft, self.state_speeds[self.state % len(self.state_speeds)])

        for col in level.get_objects():
            colx, coly = col.collide_legacy(self.pos, self.size, self.vel, ft)
            if abs(colx) > abs(xreset): 
                xreset = colx
                if col.dmg < 0:
                    self.pos.set(level.spawn)

            if abs(coly) > abs(yreset): 
                yreset = coly
                if col.dmg < 0:
                    self.pos.set(level.spawn)
        
        # adjusts height and position of crouched player
        if self.crouched:
            if self.size.y != 60:
                self.pos.y += 28
                self.size.y = 60
            
            if self.on_ground:
                Potxv /= 1.25

        else:
            if self.size.y != 88:
                self.size.y = 88
                self.pos.y -= 28

        # apply potential speeds
        self.vel.x = Potxv
        self.vel.y += Potyv
        self.vel.y += 0.0035 * ft # Gravitation

        # apply speed to position and substract possible collisions
        self.pos.y += self.vel.y * ft - yreset
        self.pos.x += Potxv * ft - xreset

        # collision rules
        if xreset != 0: 
            self.vel.x = 0

        if yreset > 0:
            if self.vel.y > 0:
                self.vel.y = 0
            self.on_ground = True
        else:
            if yreset < 0:
                self.vel.y = 0
            self.on_ground = False
            

    def input(self, ft, xv, yv=0):
        keys = pygame.key.get_pressed()
        if keys[Keybinds.JOYSTICK_DOWN]:
            self.crouched = True
        else:
            self.crouched = False

        if (keys[Keybinds.ACTION] or keys[Keybinds.JOYSTICK_UP]) and self.jump_cooldown < 1 and self.on_ground:
            self.on_ground = False
            if self.crouched == False:
                yv = -0.65
            else:
                yv = -0.35

        # accelerate player
        elif keys[Keybinds.JOYSTICK_LEFT] and keys[Keybinds.JOYSTICK_RIGHT]:
            xv = 0
        elif keys[Keybinds.JOYSTICK_RIGHT]:
            xv += 0.03
        elif keys[Keybinds.JOYSTICK_LEFT]:
            xv -= 0.03

        # running speed and air control
        if self.on_ground == True:
            if xv > 0.01:
                xv -= 0.01
            elif xv < -0.01:
                xv += 0.01
            else:
                xv = 0
        else:
            if xv > 0.005:
                xv -= 0.005
            elif xv < -0.005:
                xv += 0.005
            else:
                xv = 0

        if xv > 0.55: # max speed
            xv = 0.55
        elif xv < -0.55:
            xv = -0.55

        return xv, yv

class Game:
    def __init__(self, win, screen, camin):
        self.win = win
        self.running = True
        self.camin = camin
        self.cam = Cam(win, size=Vec2(screen[0],screen[1]))

    def run(self):
        static_objects = []
        entities = []
        # will load level1 by default
        level = Level()
        level.load()
        self.cam.set_borders(level.get_objects())

        # frametime, fps, lag = level.tick_frame(60)
        frametime = 1

        player = Player()
        player.pos = level.spawn - player.size

        render_queue = queue.PriorityQueue()
        render_count = 0

        while self.running:
            current_time = time.perf_counter()
            keys = pygame.key.get_pressed()
            for event in pygame.event.get():
                if event.type == pygame.QUIT or keys[pygame.K_ESCAPE] or keys[Keybinds.ESCAPE]:
                    self.running = False
                    # pygame.quit() # makes quitting much faster

                # elif keys[pygame.K_p]:
                #     pause = not(pause)


            player.move(frametime, level)

            self.cam.set_pos(player.pos - (self.cam.size /  2))

            # build render queue
            for i in level.get_objects():
                x, y, w, h = i.get_rect()
                relx, rely = self.cam.pos.get()
                # if (x + w - relx/64 > 0 and x - relx/64 < screen[0]) and (y + h - rely/64 > 0 and y - rely/64 < screen[1]):
                render_queue.put((2, render_count, i))
                render_count += 1

            # render background
            level.drawBackground(self.cam.size.get(), self.cam)
            # render from the queue
            for i in range(render_queue.qsize()):
                render_queue.get()[2].draw(self.cam)

            player.draw(self.cam)


            # update the screen
            pygame.display.flip()
            frametime = (time.perf_counter() - current_time) * 1000

            if frametime > 10: frametime = 10
            # frametime = 10 
            # print(frametime)

default = pygame.image.load("MissingTexture.bmp")
log = Log()

if __name__ == "__main__":
    import pygame.display, LevelReader
    pygame.display.init()
    pygame.init()
    screen = (pygame.display.Info().current_w, pygame.display.Info().current_h)
    screen = (800, 500)
    # win = pygame.display.set_mode((screen), pygame.HWSURFACE|pygame.DOUBLEBUF|pygame.FULLSCREEN|pygame.OPENGL)
    win = pygame.display.set_mode((screen), pygame.HWSURFACE|pygame.DOUBLEBUF)
    # win = pygame.display.set_mode((screen), pygame.HWSURFACE|pygame.DOUBLEBUF|pygame.FULLSCREEN, 16)
    camin = camInput.CamInput()
    camin.start()
    game = Game(win, screen, camin)
    game.run()
    camin.running = False