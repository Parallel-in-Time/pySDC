import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np


class Train_pySDC:
    """
    Interface between PyTorch and pySDC for training models.

    Attributes:
     - problem: An instantiated problem from pySDC that allows evaluating the exact solution.
                This should have the same parameters as the problem you run in pySDC later.
     - model: A PyTorch model with some neural network to train, specific to the problem
    """

    def __init__(self, problem, model, use_exact=True):
        self.problem = problem
        self.model = model
        self.use_exact = use_exact  # use exact solution in problem class or backward Euler solution

        self.model.train(True)

    def generate_initial_condition(self, t):
        return self.problem.u_exact(t)

    def generate_target_condition(self, initial_condition, t, dt):
        if self.use_exact:
            return self.problem.u_exact(t + dt)
        else:
            return self.problem.solve_system(initial_condition, dt, initial_condition, t)

    def train_model(self, initial_condition=None, t=None, dt=None, num_epochs=1000, lr=0.001):
        model = self.model

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # setup initial and target conditions
        t = torch.rand(1) if t is None else t
        dt = torch.rand(1) if dt is None else dt
        initial_condition = self.generate_initial_condition(t) if initial_condition is None else initial_condition
        target_condition = self.generate_target_condition(initial_condition, t, dt)

        # do the training
        for epoch in range(num_epochs):
            predicted_state = model(initial_condition, t, dt)
            loss = criterion(predicted_state.float(), target_condition.float())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 100 == 0 or True:
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    def plot(self, initial_condition=None, t=None, dt=None):
        t = torch.rand(1) if t is None else t
        dt = torch.rand(1) if dt is None else dt
        initial_condition = self.generate_initial_condition(t) if initial_condition is None else initial_condition
        target = self.generate_target_condition(initial_condition, t, dt)
        model_prediction = self.model(initial_condition, t, dt)

        fig, ax = plt.subplots()
        ax.plot(self.problem.xvalues, initial_condition, label='ic')
        ax.plot(self.problem.xvalues, target, label='target')
        ax.plot(self.problem.xvalues, model_prediction.detach().numpy(), label='model')
        ax.set_title(f't={t:.2e}, dt={dt:.2e}')
        ax.legend()


class HeatEquationModel(nn.Module):
    """
    Very simple model to learn the heat equation. Beware! It's too simple.
    Some machine learning expert please fix this!
    """

    def __init__(self, problem, hidden_size=64):
        self.input_size = problem.nvars * 3
        self.output_size = problem.nvars

        super().__init__()

        self.fc1 = nn.Linear(self.input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, self.output_size)

        # Initialize weights (example)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x, t, dt):
        # prepare individual tensors
        x = x.float()
        _t = torch.ones_like(x) * t
        _dt = torch.ones_like(x) * dt

        # Concatenate t and dt with the input x
        _x = torch.cat((x, _t, _dt), dim=0)

        _x = self.fc1(_x)
        _x = self.relu(_x)
        _x = self.fc2(_x)
        return _x


def train_at_collocation_nodes():
    """
    For the first proof of concept, we want to train the model specifically to the collocation nodes we use in SDC.
    If successful, the initial guess would already be the exact solution and we would need no SDC iterations.
    Alas, this neural network is too simple... We need **you** to fix it!
    """
    collocation_nodes = np.array([0.15505102572168285, 1, 0.6449489742783183]) * 1e-2

    from pySDC.playgrounds.ML_initial_guess.heat import Heat1DFDTensor

    prob = Heat1DFDTensor()
    model = HeatEquationModel(prob)
    trainer = Train_pySDC(prob, model, use_exact=True)
    for dt in collocation_nodes:
        trainer.train_model(num_epochs=50, t=0, dt=dt)
    for dt in collocation_nodes:
        trainer.plot(t=0, dt=dt)
    torch.save(model.state_dict(), 'heat_equation_model.pth')
    plt.show()


if __name__ == '__main__':
    train_at_collocation_nodes()
