import jittor as jt


class Adadelta(jt.optim.Optimizer):
    def __init__(self, params, lr=1.0, rho=0.9, eps=1e-6, weight_decay=0):
        super().__init__(params, lr)
        self.rho = rho
        self.eps = eps
        self.weight_decay = weight_decay

        # initialize required arguments for each param_groups
        for pg in self.param_groups:
            values = pg["values"] = []
            u = pg["u"] = []
            for p in pg["params"]:
                values.append(jt.zeros(p.shape, p.dtype).stop_grad())
                u.append(jt.zeros(p.shape, p.dtype).stop_grad())

    def add_param_group(self, group):
        values = group["values"] = []
        u = group["u"] = []
        for p in group["params"]:
            values.append(jt.zeros(p.shape, p.dtype).stop_grad())
            u.append(jt.zeros(p.shape, p.dtype).stop_grad())
        self.param_groups.append(group)


    def step(self, loss=None):
        if loss is not None:
            self.pre_step(loss)
        for pg in self.param_groups:
            # get arguments from each param_groups
            lr = pg.get("lr", self.lr)
            rho = pg.get("rhp", self.rho)
            eps = pg.get("eps", self.eps)
            weight_decay = pg.get("weight_decay", self.weight_decay)
            for p, g, v, u in zip(pg["params"], pg["grads"], pg["values"], pg["u"]):
                if p.is_stop_grad(): continue
                g = p * weight_decay + g
                v.update(v * rho + (1 - rho) * g * g)
                dx = jt.sqrt(u + eps) * g / jt.sqrt(v + eps)
                u.update(u * rho + (1 - rho) * dx * dx)
                p.update(p - lr * dx)

        self.zero_grad()