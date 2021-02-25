import random as rnd
import os
import neat
import math
import pygame
import visualize
import signal
import sys

NUM_GEN = 100
SIZE = 500
VISUAL = True

if VISUAL:
    pygame.init()
    pygame.display.set_caption('Petar Markovic')
    WINDOW = pygame.display.set_mode((SIZE, SIZE))

    background = pygame.Surface((SIZE, SIZE))
    background.fill(pygame.Color('#000000'))

class Point:
    def __init__(self,x,y):
        self.x = x
        self.y = y

def dist2points(a: Point, b: Point) -> float:
    return math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2)

def main(genomes, config):
    
    nets = []
    ge = []
    cells = []

    for g_id, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        cells.append(Point(200,200))
        g.fitness = 0
        ge.append(g)

    foods = [Point(rnd.randint(0,SIZE),rnd.randint(0,SIZE)) for _ in range(0,SIZE)]

    frames_without_food = 0
    while True:
        if frames_without_food > 100:
            break

        if len(foods) == 0:
            break

        if VISUAL:
            WINDOW.blit(background, (0, 0))
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False
        frames_without_food += 1
        for x, cell in enumerate(cells):
            closest_food = None
            closest_cell = None
            try:
                closest_food = min(foods, key= lambda a: dist2points(a, cell)) # Finds the closest_food food to the cell
                closest_cell = min([i for i in cells if i != cell], key= lambda a: dist2points(a, cell))
            except:
                break
            if VISUAL:
                pygame.draw.line(WINDOW,(255,0,0),(cell.x,cell.y), (closest_food.x,closest_food.y),1)
                pygame.draw.line(WINDOW,(0,0,255),(cell.x,cell.y), (closest_cell.x,closest_cell.y),1)
            output = nets[x].activate((cell.x-closest_food.x, cell.y-closest_food.y, cell.x-closest_cell.x, cell.y-closest_cell.y))
            
            cell.x += output[0] * 5
            cell.y += output[1] * 5

            if cell.y > SIZE:
                cell.y = SIZE
            elif cell.y < 0:
                cell.y = 0
            if cell.x > SIZE:
                cell.x = SIZE
            elif cell.x < 0:
                cell.x = 0

            ge[x].fitness -= 0.1
            for food in foods:
                if(dist2points(food,cell) < 5):
                    foods.remove(food)
                    ge[x].fitness += 10
                    frames_without_food = 0

        if VISUAL:    

            for food in foods:
                
                pygame.draw.circle(WINDOW,(100,255,100),(food.x,food.y),5)

            for cell in cells:
                pygame.draw.circle(WINDOW,(255,255,255),(cell.x,cell.y),5)
            
            pygame.display.update()

def run(config_path):


    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, 
                                neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

    p = neat.Population(config)

    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    def signal_handler(sig, frame):
        visualize.plot_stats(stats, ylog=False, view=True)
        visualize.plot_species(stats, view=True)
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    winner = p.run(main,NUM_GEN)
    visualize.draw_net(config, winner, True, node_names)
    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)
    drawData(stats)


if __name__ == '__main__':

    os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin'

    local_directory = os.path.dirname(__file__)
    config_path = os.path.join(local_directory, "config-feedforward.txt")
    run(config_path)