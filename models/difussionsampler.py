import torch
import numpy as np


def get_named_beta_schedule(num_diffusion_steps, p=0.1):
    """
    Get a pre-defined beta schedule for the given name.
    """
    return [
        (n - 1) ** p / (num_diffusion_steps - 1) ** (p - 1)
        for n in range(1, num_diffusion_steps + 1)
    ]


def get_named_eta_schedule(num_diffusion_steps, kappa, beta_schedule):
    """
    Get a pre-defined eta schedule for the given name.
    """
    eta_N = 0.999
    eta_0 = min(0.001, (0.04 / kappa) ** 2)

    b0 = np.exp(1 / (2 * (num_diffusion_steps - 1)) * np.log(eta_N / eta_0))

    sqrt_eta = [eta_0**0.5]
    for n in range(2, num_diffusion_steps):
        sqrt_eta.append(sqrt_eta[0] * b0 ** beta_schedule[n - 1])
    return sqrt_eta + [eta_N**0.5]


class DiffusionSampler(object):
    """
    Utilities for training and sampling diffusion models.

    :param sqrt_etas: a 1-D numpy array of etas for each diffusion timestep,
                starting at T and going to 1.
    :param kappa: a scaler controling the variance of the diffusion kernel
    :param model_mean_type: a ModelMeanType determining what the model outputs.
    :param loss_type: a LossType determining the loss function to use.
                              model so that they are always scaled like in the
                              original paper (0 to 1000).
    :param scale_factor: a scaler to scale the latent code
    :param sf: super resolution factor
    """

    def __init__(
        self,
        num_diffusion_steps,
        kappa,
        device,
    ):

        self.num_diffusion_steps = num_diffusion_steps
        self.kappa = kappa

        if num_diffusion_steps == 1:
            self.sqrt_etas_schedule = [0]
        else:
            self.beta_schedule = get_named_beta_schedule(num_diffusion_steps)
            self.sqrt_etas_schedule = get_named_eta_schedule(
                num_diffusion_steps, kappa, self.beta_schedule
            )
        print("sqrt_etas:")
        print(self.sqrt_etas_schedule)
        self.sqrt_etas = torch.tensor(self.sqrt_etas_schedule)
        self.device = device

    def q_sample(self, x_0, e, n, mean=False):
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_n | x_0, y_0).

        :param x_0: the initial data batch NxT.
        :param e (y0-x0): the [N x C x ...] tensor of differences to the degraded inputs.
        :param n: the number of diffusion steps. starting from 1.
        :return: A noisy version of x_0, a sample from q(x_n | x_0, y_0)
        """
        index_value = (
            torch.tensor([self.sqrt_etas[n_b - 1] for n_b in n])
            .unsqueeze(1)
            .unsqueeze(2)
            .float()
            .to(self.device)
        )

        return (
            index_value**2 * e
            + x_0
            + index_value * self.kappa * torch.randn_like(x_0).to(self.device)
        )
