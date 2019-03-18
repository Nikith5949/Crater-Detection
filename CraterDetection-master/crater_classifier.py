import cPickle
import theano
import theano.tensor as T
import numpy as np

CRATER = 1
NOT_CRATER = 0

class CraterClassifier(object):

    def __init__(self, net):
        """ Makes an object that takes a list of images and classifies
            them as either craters or non-craters. """
        self.network = net


    def shared(self, data):
        """Place the data into shared variables.  This allows Theano to copy
        the data to the GPU, if one is available.
        """
        shared_x = theano.shared(
            np.asarray(data, dtype=theano.config.floatX), borrow=True)
        return shared_x

    def get_classifications(self, images):
        """ Perform classifications on image data set. Adds to hitlist. """
        self.images = self.shared(images)
        self.num_imgs = self.images.shape[0].eval()
        i = T.lscalar()
        self.classify_fn = theano.function(
            [i], self.network.layers[-1].y_out,
            givens={
                self.network.x:
                self.images[i*self.network.mini_batch_size: (i+1)*self.network.mini_batch_size]
            })
        classifications = []
        for i in range(self.num_imgs):
            classifications.append(self.classify_fn(i)[0])
        return classifications

class CraterHitList(object):

    def __init__(self,original_width,swz,gts):
        self.original_width = original_width
        self.swz = swz
        self.GTs = gts
        self.TPs = {}
        self.FPs = []

    def add_hits(self, query):
        images, centerpoints, scales, classifications = query
        n = len(images)
        tp = fp = 0
        for i in range(n):
            if classifications[i] == CRATER:
                hit = Hit(centerpoints[i], self.swz/2, scales[i])
                found = self.GTs.found_crater(hit)
                if not found:
                    fp += 1
                    self.add_FP(hit)
                else:
                    for crater_ID in found:
                        tp += 1
                        self.add_TP(crater_ID,hit)
        return (tp, fp)

    def add_TP(self, id, hit):
        if id not in self.TPs:
            self.TPs[id] = []
        self.TPs[id].append(hit)

    def add_FP(self, hit):
        self.FPs.append(hit)

    def get_TP_hits(self):
        tp_hits = []
        for crater_hits in self.TPs.values():
            tp_hits += crater_hits
        return tp_hits

    def get_FP_hits(self):
        return self.FPs

class Hit(object):
    def __init__(self, cp, radius, scale):
        self.x = cp[0]
        self.y = cp[1]
        self.radius = radius
        self.scale = scale


if __name__ == '__main__':
    import crater_loader as cl
    network_pickle = 'Pickle_Stash/ELU-ntwk-e0-val0.9709-tst0.9643.pkl'
    dataset= "101x101.pkl"
    image_size = 101
    print "Loading network..."
    classifier = cPickle.load(open(network_pickle))
    print "Getting Labels...."
    images, labels = cPickle.load(open(dataset))[2]
    print images[:5]
    print "Making classifier..."
    cc = CraterClassifier(network_pickle, images)
    print "Making classifications with network..."
    results = cc.get_classifications()
    # for tup in zip(results, labels):
    #     print tup
    print results
