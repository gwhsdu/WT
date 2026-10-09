class EarlyStopper:
    def __init__(self, patience=100, delta=0.001, mode='min', path='best_model.pt'):
        self.patience = patience
        self.delta = delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.path = path

        if mode not in ['min', 'max']:
            raise ValueError("mode must be 'min' or 'max'")

    def __call__(self, current_score, model):
        if self.best_score is None:
            self.best_score = current_score
            # self.save_model(model)
        else:
            if self.mode == 'min':
                improve = current_score < (self.best_score - self.delta)
            else:
                improve = current_score > (self.best_score + self.delta)

            if improve:
                self.best_score = current_score
                # self.save_model(model)
                self.counter = 0
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True