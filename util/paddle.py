import pygame
BLACK = (0, 0, 0)
WHITE = (255,255,255)

class Paddle(pygame.sprite.Sprite):
    speed = 15
    def __init__(self, color, width, height):
        super().__init__()

        #set
        self.image = pygame.Surface([width, height])
        self.image.fill(BLACK)
        self.image.set_colorkey(BLACK)

        #draw
        pygame.draw.rect(self.image, color, [0, 0, width, height])

        #get rect
        self.rect = self.image.get_rect()

    def move_up(self, pixels):
        self.rect.y -= pixels
        if self.rect.y < 0:
            self.rect.y = 0

    def move_down(self, pixels):
        self.rect.y += pixels
        if self.rect.y > 400:
            self.rect.y = 400
