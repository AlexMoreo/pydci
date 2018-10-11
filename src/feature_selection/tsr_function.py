import math

def information_gain(cell):
    def ig_factor(p_tc, p_t, p_c):
        den = p_t * p_c
        if den != 0.0 and p_tc != 0:
            return p_tc * math.log(p_tc / den, 2)
        else:
            return 0.0

    return ig_factor(cell.p_tp(), cell.p_f(), cell.p_c()) + ig_factor(cell.p_fp(), cell.p_f(), cell.p_not_c()) \
           + ig_factor(cell.p_fn(), cell.p_not_f(), cell.p_c()) + ig_factor(cell.p_tn(), cell.p_not_f(), cell.p_not_c())


class ContTable:
    def __init__(self, tp=0, tn=0, fp=0, fn=0):
        self.tp=tp
        self.tn=tn
        self.fp=fp
        self.fn=fn

    def get_d(self): return self.tp + self.tn + self.fp + self.fn

    def get_c(self): return self.tp + self.fn

    def get_not_c(self): return self.tn + self.fp

    def get_f(self): return self.tp + self.fp

    def get_not_f(self): return self.tn + self.fn

    def p_c(self): return (1.0*self.get_c())/self.get_d()

    def p_not_c(self): return 1.0-self.p_c()

    def p_f(self): return (1.0*self.get_f())/self.get_d()

    def p_not_f(self): return 1.0-self.p_f()

    def p_tp(self): return (1.0*self.tp) / self.get_d()

    def p_tn(self): return (1.0*self.tn) / self.get_d()

    def p_fp(self): return (1.0*self.fp) / self.get_d()

    def p_fn(self): return (1.0*self.fn) / self.get_d()

    def tpr(self):
        c = 1.0*self.get_c()
        return self.tp / c if c > 0.0 else 0.0

    def fpr(self):
        _c = 1.0*self.get_not_c()
        return self.fp / _c if _c > 0.0 else 0.0

