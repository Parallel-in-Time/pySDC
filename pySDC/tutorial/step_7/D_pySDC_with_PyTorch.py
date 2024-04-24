import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pySDC.playgrounds.ML_initial_guess.ml_heat import HeatEquationModel, Train_pySDC
from pySDC.playgrounds.ML_initial_guess.heat import Heat1DFDTensor


def train_at_collocation_nodes():
    """
    For the first proof of concept, we want to train the model specifically to the collocation nodes we use in SDC.
    If successful, the initial guess would already be the exact solution and we would need no SDC iterations.

    What we find is that we can train the network to predict the solution to one very specific problem rather well.
    See the error during training for what happens when we ask the network to solve for exactly what it just trained.
    However, if we train for something else, i.e. solving to a different step size in this case, we can only use the
    model to predict the solution of what it's been trained for last and it loses the ability to solve for previously
    learned things. This is solely because we chose an overly simple model that is unsuitable to the task at hand and
    is likely easily solved with a bit of patience. This is just a demonstration of the interface between pySDC and
    PyTorch. If you want to do a project with this, feel free to take this as a starting point and do things that
    actually do something!

    The output shows the training loss during training and, after each of three training sessions is complete, the error
    of the prediction with the current state of the network. To demonstrate the forgetfulness, we finally print the
    error of all learned predictions after training is complete.
    """
    out = ''
    errors_mid_training = []
    errors_post_training = []

    # instantiate the pySDC problem and a model for PyTorch
    problem = Heat1DFDTensor()
    model = HeatEquationModel(problem)

    # setup neural network
    lr = 0.001
    num_epochs = 250
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # setup initial conditions
    t = 0
    initial_condition = problem.u_exact(t)

    # train the model to predict the solution at certain collocation nodes
    collocation_nodes = np.array([0.15505102572168285, 0.6449489742783183, 1]) * 1e-2
    for dt in collocation_nodes:

        # get target condition from implicit Euler step
        target_condition = problem.solve_system(initial_condition, dt, initial_condition, t)

        # do the training
        for epoch in range(num_epochs):
            predicted_state = model(initial_condition, t, dt)
            loss = criterion(predicted_state.float(), target_condition.float())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 50 == 0:
                out += f'Training for {dt=:.2e}: Epoch [{epoch+1:4d}/{num_epochs:4d}], Loss: {loss.item():.4e}\n'

        # evaluate model to compute error
        model_prediction = model(initial_condition, t, dt)
        errors_mid_training += [abs(target_condition - model_prediction)]
        out += f'Error of prediction at {dt:.2e} during training: {abs(target_condition-model_prediction):.2e}\n'

    # compare model and problem
    for dt in collocation_nodes:
        target_condition = problem.solve_system(initial_condition, dt, initial_condition, t)
        model_prediction = model(initial_condition, t, dt)
        errors_post_training += [abs(target_condition - model_prediction)]
        out += f'Error of prediction at {dt:.2e} after training: {abs(target_condition-model_prediction):.2e}\n'

    print(out)
    with open('data/step_7_D_out.txt', 'w') as file:
        file.write(out)

    # test that the training went as expected
    assert np.greater([1e-2, 1e-4, 1e-5], errors_mid_training).all(), 'Errors during training are larger than expected'
    assert np.greater([1e0, 1e0, 1e-5], errors_post_training).all(), 'Errors after training are larger than expected'

    # save the model to use it throughout pySDC
    torch.save(model.state_dict(), 'data/heat_equation_model.pth')


if __name__ == '__main__':
    train_at_collocation_nodes()
