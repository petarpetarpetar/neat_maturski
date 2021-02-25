class Point:
    def __init__(self,x,y):
        self.x = x
        self.y = y
        

class Worm:
    def __init__(self,a,b,c):
        self.cells = [a,b,c]
    

def dist2points(a: Point, b: Point) -> float:
    return math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2)