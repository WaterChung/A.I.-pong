import pygame, sys, copy
from util.paddle import Paddle
from util.ball import Ball
from util.AI import NeuralNetwork
pygame.init()

MAX_RANDOM_GENOME = 50

#colors
BLACK = [0, 0, 0]
WHITE = [255,255,255]

#window
size = (700,500)
screen = pygame.display.set_mode(size)
pygame.display.set_caption("Pong")
carryOn = True
clock = pygame.time.Clock()

#paddles
paddleA = Paddle(WHITE, 10, 100)
paddleA.rect.x = 20
paddleA.rect.y = 200

paddleB = Paddle(WHITE, 10, 100)
paddleB.rect.x = 670
paddleB.rect.y = 200

#ball
ball = Ball(WHITE, 10, 10)
ball.rect.x = 345
ball.rect.y = 195

#sprites
all_sprites_list = pygame.sprite.Group()
all_sprites_list.add(paddleA)
all_sprites_list.add(paddleB)
all_sprites_list.add(ball)

#setup neural network
neural_networkA = NeuralNetwork()
best_genome_input_weightsA = []
best_genome_hidden_weightsA = []
genome_countA = 1

neural_networkB = NeuralNetwork()
best_genome_input_weightsB = []
best_genome_hidden_weightsB = []
genome_countB = 1

#controllers
moving_leftA = False
moving_rightA = False
move_directionA = "standby"

moving_leftB = False
moving_rightB = False
move_directionB = "standby"

#score
scoreA = 0
scoreB = 0
hidden_scoreA = 0
score_recordA = 0
hidden_scoreB = 0
score_recordB = 0

#main
while carryOn:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            carryOn = False

    all_sprites_list.update()

    #A.I directions
    ai_resultB = neural_networkB.get_output(ball.rect.x, paddleB.rect.x) #get result of the neural
    if ai_resultB >= 0.6: #this can be any value of 0.0 to 1.0
        move_directionB = "right"
    elif ai_resultB <= 0.4:
        move_directionB = "left"
    else:
        move_directionB = "standby"

    ai_resultA = neural_networkA.get_output(ball.rect.x, paddleA.rect.x)
    if ai_resultA >= 0.6: #this can be any value of 0.0 to 1.0
        move_directionA = "right"
    elif ai_resultA <= 0.4:
        move_directionA = "left"
    else:
        move_directionA = "standby"
    #A.I movements
    if (move_directionA == "right") and (paddleA.rect.y < 700 - 10):
        paddleA.move_up(10)
    elif (move_directionA == "left") and (paddleA.rect.y > 0):
        paddleA.move_down(10)

    if (move_directionB == "right") and (paddleB.rect.y < 700 - 10):
        paddleB.move_up(10)
    elif (move_directionB == "left") and (paddleB.rect.y > 0):
        paddleB.move_down(10)
    #ball check
    if ball.rect.x >= 690:
        scoreA += 1
        ball.velocity[0] = -ball.velocity[0]
    if ball.rect.x <= 0:
        scoreB += 1
        ball.velocity[0] = -ball.velocity[0]
    if ball.rect.y > 490:
        ball.velocity[1] = -ball.velocity[1]
        hidden_scoreB += 1
    if ball.rect.y < 0:
        ball.velocity[1] = -ball.velocity[1]
        hidden_scoreA += 1

    if pygame.sprite.collide_mask(ball, paddleA) or pygame.sprite.collide_mask(ball, paddleB):
        ball.bounce()

        if hidden_scoreB > score_recordB:
            best_genome_input_weightsB = copy.deepcopy(neural_networkB.input_weights)
            best_genome_hidden_weightsB = copy.deepcopy(neural_networkB.hidden_weights)
            score_record = hidden_scoreB

        if (genome_countB >= MAX_RANDOM_GENOME) and (score_recordB > 0):
            neural_networkB.input_weights = copy.deepcopy(best_genome_input_weightsB)
            neural_networkB.hidden_weights = copy.deepcopy(best_genome_hidden_weightsB)
            neural_networkB.make_mutation()
            print("New mutation: " + str(neural_networkB.input_weights) + " - " + str(neural_networkB.hidden_weights))
        else:
            neural_networkB.generate_weights()
            print("New genome: " + str(neural_networkB.input_weights) + " - " + str(neural_networkB.hidden_weights))

        hidden_scoreB = 0
        genome_countB += 1


        if hidden_scoreA > score_recordA:
            best_genome_input_weightsA = copy.deepcopy(neural_networkA.input_weights)
            best_genome_hidden_weightsA = copy.deepcopy(neural_networkA.hidden_weights)
            score_recordA = hidden_scoreA

        if (genome_countA >= MAX_RANDOM_GENOME) and (score_recordA > 0):
            neural_networkA.input_weights = copy.deepcopy(best_genome_input_weightsA)
            neural_networkA.hidden_weights = copy.deepcopy(best_genome_hidden_weightsA)
            neural_networkA.make_mutation()
            print("New mutation: " + str(neural_networkA.input_weights) + " - " + str(neural_networkA.hidden_weights))
        else:
            neural_networkA.generate_weights()
            print("New genome: " + str(neural_networkA.input_weights) + " - " + str(neural_networkA.hidden_weights))

        hidden_scoreA = 0
        genome_countA += 1

    screen.fill(BLACK)
    pygame.draw.line(screen, WHITE, [349, 0], [349,500], 5)
    all_sprites_list.draw(screen)


    #display the score
    font = pygame.font.Font(None, 70)
    text = font.render(str(scoreA), 1, WHITE)
    screen.blit(text, (250,10))
    text = font.render(str(scoreB), 1, WHITE)
    screen.blit(text, (490,10))
    pygame.display.flip()
    clock.tick(60)

pygame.quit()