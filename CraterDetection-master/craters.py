"5-12-18 Garrett Alston"

from math import pow
from crater_classifier import Hit

class Circle(object):
    def __init__(self, center, radius):
        self.x = float(center[0])
        self.y = float(center[1])
        self.radius = float(radius)
    def intersects(self, circle):
        distSq = pow(self.x - circle.x, 2) + pow(self.y - circle.y, 2);
        radSumSq = pow(self.radius + circle.radius, 2);
        if (distSq > radSumSq):
            return False
        else:
            return True

class Crater(object):

    def __init__(self, center, radius):
        self.circle = Circle(center, radius)

    def is_hit(self, hit):
        x = float(hit.x) / hit.scale
        y = float(hit.y) / hit.scale
        radius = (float(hit.radius) / hit.scale) * .5
        range = Circle([x,y], radius)
        return self.circle.intersects(range)

class CraterList(object):

    def __init__(self):
        self.craters = []

    def size(self):
        return len(self.craters)

    def get_crater(self, i):
        return self.craters[i]

    def add(self, center, radius):
        self.craters.append(Crater(center, radius))

    def found_crater(self, hit):
        found = []
        for id in range(self.size()):
            if self.craters[id].is_hit(hit):
                found.append(id)
        if len(found) == 0:
            return False
        return found


if __name__ == '__main__':

    print "Testing Crater data structure"
    crater = Crater([0,0], 8)
    p_in = [1,1]
    p_out = [40,40]
    print crater.is_hit(p_in)
    print crater.is_hit(p_out)

    print "\nTesting CraterList data structure"
    craters = CraterList()
    #craters.add([0,0], 10)
    craters.add([0,0], 8)
    craters.add([0,0], 4)
    craters.add([0,0], 2)
    print craters.found_crater(p_in, 1)
    print craters.found_crater(p_out, 1)
    #craters.add([0,0], 100)
    scale = 2
    for s in range(-4, 8, 1):
        true_scale = pow(scale, -s)
        # x = float(p_in[0]) * true_scale
        # y = float(p_in[1]) * true_scale
        print "[%s, %s] scaled from %s in craters: %s" \
                    % (1,1,true_scale,craters.found_crater(p_in, true_scale))
