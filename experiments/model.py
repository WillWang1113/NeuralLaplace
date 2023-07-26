###########################
# Neural Laplace: Learning diverse classes of differential equations in the Laplace domain
# Author: Samuel Holt
###########################
import logging

import torch
from torch import nn
from NeuralLaplace.torchlaplace import laplace_reconstruct
import NeuralLaplace.torchlaplace.inverse_laplace
from encoders import CNNEncoder, DNNEncoder, ReverseGRUEncoder, BiGRUEncoder

logger = logging.getLogger()

# # ? Should be independent with s
# class SphereSurfaceModel(nn.Module):
#     # C^{b+k} -> C^{bxd} - In Riemann Sphere Co ords : b dim s reconstruction terms, k is latent encoding dimension, d is output dimension
#     def __init__(self,
#                  s_dim,
#                  output_dim,
#                  latent_dim,
#                  hidden_units=64,
#                  **kwargs):
#         super(SphereSurfaceModel, self).__init__()
#         self.s_dim = s_dim
#         self.output_dim = output_dim
#         self.latent_dim = latent_dim
#         self.linear_tanh_stack = nn.Sequential(
#             nn.Linear(s_dim * 2 + latent_dim, hidden_units),
#             nn.Tanh(),
#             nn.Linear(hidden_units, hidden_units),
#             nn.Tanh(),
#             nn.Linear(hidden_units, (s_dim) * 2 * output_dim),
#         )

#         # for m in self.linear_tanh_stack.modules():
#         #     if isinstance(m, nn.Linear):
#         #         nn.init.xavier_uniform_(m.weight)

#         # TODO: manual set the phi min max
#         # self.phi_max = 2 * (torch.atan(torch.tensor(3.0)) - torch.pi / 4.0)
#         self.phi_max = torch.pi / 2.0
#         # self.phi_min = 2 * (torch.atan(torch.tensor(20.0)) - torch.pi / 4.0)
#         self.phi_min = -torch.pi / 2.0
#         self.phi_scale = self.phi_max - self.phi_min

#         self.theta_max = torch.pi
#         # self.theta_min = 2 * (torch.atan(torch.tensor(20.0)) - torch.pi / 4.0)
#         self.theta_min = -torch.pi
#         self.theta_scale = self.theta_max - self.theta_min
#         self.nfe = 0

#     def forward(self, i):
#         # Take in initial conditon p and the Rieman representation
#         self.nfe += 1
#         out = self.linear_tanh_stack(
#             i.view(-1, self.s_dim * 2 + self.latent_dim)).view(
#                 -1, 2 * self.output_dim, self.s_dim)
#         # theta = nn.Tanh()(
#         #     out[:, :self.output_dim, :]) * self.theta_scale / 2.0 + self.theta_min + self.theta_scale / 2.0
#         theta = nn.Tanh()(
#             out[:, :self.output_dim, :]) * torch.pi  # From - pi to + pi
#         # phi = (nn.Tanh()(out[:, self.output_dim:, :]) * self.phi_scale / 2.0 +
#         #        self.phi_min + self.phi_scale / 2.0)  # Form -pi / 2 to + pi / 2
#         phi = (nn.Tanh()(out[:, self.output_dim:, :]) * self.phi_scale / 2.0 -
#                torch.pi / 2.0 + self.phi_scale / 2.0
#                )  # Form -pi / 2 to + pi / 2
#         return theta, phi


# ? Should be independent with s
class SphereSurfaceModel(nn.Module):
    # C^{b+k} -> C^{bxd} - In Riemann Sphere Co ords : b dim s reconstruction terms, k is latent encoding dimension, d is output dimension
    def __init__(self,
                 s_dim,
                 output_dim,
                 latent_dim,
                 out_timesteps,
                 include_s_recon_terms=True,
                 hidden_units=64):
        super(SphereSurfaceModel, self).__init__()
        self.s_dim = s_dim
        self.output_dim = output_dim
        self.latent_dim = latent_dim
        self.include_s_recon_terms = include_s_recon_terms
        print("include s recon terms:", include_s_recon_terms)
        dim_in = (2 * s_dim +
                  latent_dim) if include_s_recon_terms else (2 + latent_dim)
        dim_out = (2 * output_dim *
                   s_dim) if include_s_recon_terms else (2 * output_dim)
        self.dim_in = dim_in
        self.linear_tanh_stack = nn.Sequential(
            nn.Linear(dim_in, hidden_units),
            # nn.BatchNorm1d(hidden_units),
            nn.Tanh(),
            nn.Linear(hidden_units, hidden_units),
            # nn.BatchNorm1d(hidden_units),
            nn.Tanh(),
            nn.Linear(hidden_units, dim_out),
        )
        self.out_timesteps = out_timesteps
        self.divide_point = (
            self.output_dim *
            self.s_dim) if self.include_s_recon_terms else self.output_dim

        self.phi_max = torch.pi / 2.0
        self.phi_min = -torch.pi / 2.0
        self.phi_scale = self.phi_max - self.phi_min

        self.theta_max = torch.pi
        self.theta_min = -torch.pi
        self.theta_scale = self.theta_max - self.theta_min
        self.nfe = 0

    def forward(self, i):
        # Take in initial conditon p and the Rieman representation
        # If include_s_recon_terms: inputs shape: [batchsize, 2 * s_dim + latent_dim]
        # else                      inputs shape: [batchsize, s_dim, 2 + latent_dim]
        self.nfe += 1
        out = self.linear_tanh_stack(i.view(-1, self.dim_in))

        theta = nn.Tanh()(
            out[..., :self.divide_point]) * torch.pi  # From - pi to + pi
        phi = (nn.Tanh()(out[..., self.divide_point:]) * self.phi_scale / 2.0 -
               torch.pi / 2.0 + self.phi_scale / 2.0
               )  # Form -pi / 2 to + pi / 2
        theta = theta.view(i.shape[0], 1, -1).repeat(1, self.out_timesteps, 1)
        phi = phi.view(i.shape[0], 1, -1).repeat(1, self.out_timesteps, 1)
        return theta, phi


class SurfaceModel(nn.Module):
    # C^{b+k} -> C^{bxd} - In Riemann Sphere Co ords : b dim s reconstruction terms, k is latent encoding dimension, d is output dimension
    def __init__(self, s_dim, output_dim, latent_dim, hidden_units=64):
        super(SurfaceModel, self).__init__()
        self.s_dim = s_dim
        self.output_dim = output_dim
        self.latent_dim = latent_dim
        self.linear_tanh_stack = nn.Sequential(
            nn.Linear(s_dim * 2 + latent_dim, hidden_units),
            nn.Tanh(),
            nn.Linear(hidden_units, hidden_units),
            nn.Tanh(),
            nn.Linear(hidden_units, (s_dim) * 2 * output_dim),
        )

        # for m in self.linear_tanh_stack.modules():
        #     if isinstance(m, nn.Linear):
        #         nn.init.xavier_uniform_(m.weight)
        self.nfe = 0

    def forward(self, i):
        self.nfe += 1
        out = self.linear_tanh_stack(
            i.view(-1, self.s_dim * 2 + self.latent_dim)).view(
                -1, 2 * self.output_dim, self.s_dim)
        real = out[:, :self.output_dim, :]
        imag = out[:, self.output_dim:, :]
        return real, imag


class MyNeuralLaplace(nn.Module):

    def __init__(self,
                 input_dim,
                 output_dim,
                 latent_dim=2,
                 hidden_units=64,
                 s_recon_terms=33,
                 use_sphere_projection=True,
                 encode_obs_time=True,
                 include_s_recon_terms=True,
                 ilt_algorithm="fourier",
                 device="cpu",
                 encoder="dnn",
                 input_timesteps=None,
                 output_timesteps=None,
                 start_k=0):
        super(MyNeuralLaplace, self).__init__()
        self.latent_dim = latent_dim

        if encoder == "cnn":
            self.encoder = CNNEncoder(input_dim,
                                      latent_dim,
                                      hidden_units // 2,
                                      encode_obs_time,
                                      timesteps=input_timesteps)
        elif encoder == "dnn":
            self.encoder = DNNEncoder(input_dim,
                                      latent_dim,
                                      hidden_units // 2,
                                      encode_obs_time,
                                      timesteps=input_timesteps)
        elif encoder == "rnn":
            self.encoder = ReverseGRUEncoder(input_dim, latent_dim,
                                             hidden_units // 2,
                                             encode_obs_time)
        else:
            raise ValueError("encoders only include dnn, cnn and rnn")

        self.use_sphere_projection = use_sphere_projection
        self.output_dim = output_dim
        self.start_k = start_k
        self.ilt_algorithm = ilt_algorithm
        self.include_s_recon_terms = include_s_recon_terms
        self.s_recon_terms = s_recon_terms
        if use_sphere_projection:
            self.laplace_rep_func = SphereSurfaceModel(
                s_dim=s_recon_terms,
                include_s_recon_terms=include_s_recon_terms,
                output_dim=output_dim,
                latent_dim=latent_dim,
                out_timesteps=output_timesteps)
        else:
            self.laplace_rep_func = SurfaceModel(s_recon_terms, output_dim,
                                                 latent_dim)
        NeuralLaplace.torchlaplace.inverse_laplace.device = device

    def forward(self, observed_data, observed_tp, tp_to_predict):
        # trajs_to_encode : (N, T, D) tensor containing the observed values.
        # tp_to_predict: Is the time to predict the values at.
        p = self.encoder(observed_data, observed_tp)
        out = laplace_reconstruct(
            self.laplace_rep_func,
            p,
            tp_to_predict,
            ilt_reconstruction_terms=self.s_recon_terms,
            # recon_dim=self.latent_dim,
            recon_dim=self.output_dim,
            use_sphere_projection=self.use_sphere_projection,
            include_s_recon_terms=self.include_s_recon_terms,
            ilt_algorithm=self.ilt_algorithm,
            options={"start_k": self.start_k})
        # out = self.output_dense(out)
        # out = out.reshape(observed_data.shape[0], -1, self.output_dim)
        return out


class GeneralNeuralLaplace(nn.Module):

    def __init__(self,
                 input_dim,
                 output_dim,
                 input_timesteps,
                 output_timesteps,
                 latent_dim=2,
                 hidden_units=64,
                 s_recon_terms=33,
                 use_sphere_projection=True,
                 include_s_recon_terms=True,
                 encode_obs_time=True,
                 encoder="dnn",
                 ilt_algorithm="fourier",
                 device="cpu",
                 method="single",
                 **kwargs):
        super(GeneralNeuralLaplace, self).__init__()

        if method == "single" and isinstance(s_recon_terms, int):
            self.model = MyNeuralLaplace(
                input_dim,
                output_dim,
                latent_dim=latent_dim,
                hidden_units=hidden_units,
                s_recon_terms=s_recon_terms,
                use_sphere_projection=use_sphere_projection,
                encode_obs_time=encode_obs_time,
                include_s_recon_terms=include_s_recon_terms,
                ilt_algorithm=ilt_algorithm,
                device=device,
                encoder=encoder,
                input_timesteps=input_timesteps,
                output_timesteps=output_timesteps,
                start_k=kwargs.get("start_k", 0))
        elif method == "hierarchical" and isinstance(s_recon_terms, list):
            self.model = HierarchicalNeuralLaplace(
                input_dim=input_dim,
                output_dim=output_dim,
                input_timesteps=input_timesteps,
                output_timesteps=output_timesteps,
                latent_dim=latent_dim,
                hidden_units=hidden_units,
                s_recon_terms=s_recon_terms,
                use_sphere_projection=use_sphere_projection,
                include_s_recon_terms=include_s_recon_terms,
                encode_obs_time=encode_obs_time,
                ilt_algorithm=ilt_algorithm,
                encoder=encoder,
                device=device,
                pass_raw=kwargs.get("pass_raw", False))

        else:
            raise ValueError(
                "Neural Laplace method can only be 'single' or 'hierarchical'."
            )
        self.method = method
        self.loss_fn = torch.nn.MSELoss()

    def _get_loss(self, dl):
        cum_loss = 0
        cum_samples = 0
        for batch in dl:
            if self.method == "single":
                preds = self.model(batch["observed_data"],
                                   batch["observed_tp"],
                                   batch["tp_to_predict"])
            else:
                preds, _ = self.model(batch["observed_data"],
                                      batch["observed_tp"],
                                      batch["tp_to_predict"])
                # cum_loss += recon_loss
            cum_loss += self.loss_fn(torch.flatten(preds),
                                     torch.flatten(batch["data_to_predict"])
                                     ) * batch["observed_data"].shape[0]
            cum_samples += batch["observed_data"].shape[0]
        mse = cum_loss / cum_samples
        return mse

    def training_step(self, batch):
        if self.method == "single":
            preds = self.model(batch["observed_data"], batch["observed_tp"],
                               batch["tp_to_predict"])
            recon_loss = 0
        else:
            preds, recon_loss = self.model(batch["observed_data"],
                                           batch["observed_tp"],
                                           batch["tp_to_predict"])
        # print(preds.shape)
        # print(batch["data_to_predict"].shape)
        loss = self.loss_fn(torch.flatten(preds),
                            torch.flatten(
                                batch["data_to_predict"])) + recon_loss
        return loss

    @torch.no_grad()
    def validation_step(self, dlval):
        self.model.eval()
        mse = self._get_loss(dlval)
        return mse, mse

    @torch.no_grad()
    def test_step(self, dltest):
        self.model.eval()
        mse = self._get_loss(dltest)
        return mse, mse

    @torch.no_grad()
    def predict(self, dl):
        self.model.eval()
        predictions, trajs = [], []
        for batch in dl:
            if self.method == "single":
                preds = self.model(batch["observed_data"],
                                   batch["observed_tp"],
                                   batch["tp_to_predict"])
            else:
                preds, _ = self.model(batch["observed_data"],
                                      batch["observed_tp"],
                                      batch["tp_to_predict"])
            predictions.append(preds)
            if batch["mode"] == "extrap":
                trajs.append(batch["data_to_predict"])
                # trajs.append(
                #     torch.cat(
                #         (batch["observed_data"], batch["data_to_predict"]),
                #         axis=1))
            else:
                trajs.append(batch["data_to_predict"])
        return torch.cat(predictions, 0), torch.cat(trajs, 0)

    def encode(self, dl):
        encodings = []
        for batch in dl:
            encodings.append(
                self.model.encode(batch["observed_data"],
                                  batch["observed_tp"]))
        return torch.cat(encodings, 0)

    # def _get_and_reset_nfes(self):
    #     """Returns and resets the number of function evaluations for model."""
    #     iteration_nfes = self.model.laplace_rep_func.nfe
    #     self.model.laplace_rep_func.nfe = 0
    #     return iteration_nfes


class Interpolate(nn.Module):

    def __init__(self, size, mode):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.size = size
        self.mode = mode

    def forward(self, x):
        x = self.interp(x, size=self.size, mode=self.mode)
        return x


class Transpose(nn.Module):

    def __init__(self, dim0, dim1):
        super(Transpose, self).__init__()
        self.transpose = torch.transpose
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):
        x = self.transpose(x, self.dim0, self.dim1)
        return x


def AIblock(avg_kernalsize, avg_stride, interp_size, interp_mode):
    return nn.Sequential(Transpose(1, 2),
                         nn.AvgPool1d(avg_kernalsize, avg_stride),
                         Interpolate(interp_size, interp_mode),
                         Transpose(1, 2))


class HierarchicalNeuralLaplace(nn.Module):

    def __init__(self,
                 input_dim,
                 output_dim,
                 input_timesteps,
                 output_timesteps,
                 latent_dim=2,
                 hidden_units=64,
                 s_recon_terms=33,
                 use_sphere_projection=True,
                 include_s_recon_terms=True,
                 encode_obs_time=False,
                 ilt_algorithm="fourier",
                 encoder="dnn",
                 device="cpu",
                 pass_raw=False):
        super(HierarchicalNeuralLaplace, self).__init__()
        self.input_timesteps = input_timesteps
        self.output_timesteps = output_timesteps
        self.use_sphere_projection = use_sphere_projection
        self.include_s_recon_terms = include_s_recon_terms
        self.ilt_algorithm = ilt_algorithm
        self.output_dim = output_dim
        self.pass_raw = pass_raw
        # print(self.input_timesteps)
        # print(self.output_timesteps)

        # self.aiblk_list = nn.ModuleList([
        #     AIblock(12, 12, input_timesteps, "linear"),
        #     AIblock(6, 6, input_timesteps, "linear"),
        #     AIblock(2, 2, input_timesteps, "linear")
        # ])
        # s_recon_terms_list = [s_recon_terms * i for i in [1, 2, 3]]
        # for i in range(len(s_recon_terms_list)):
        #     if s_recon_terms_list[i] % 2 == 0:
        #         s_recon_terms_list[i] += 1
        # print(s_recon_terms_list)
        start_ks = [0]
        for i in range(len(s_recon_terms) - 1):
            start_ks.append(start_ks[-1] + s_recon_terms[i])
        print(start_ks)
        print(s_recon_terms)
        self.start_ks = start_ks
        self.s_recon_terms_list = s_recon_terms

        if encoder == "cnn":
            encoder = CNNEncoder
        elif encoder == "dnn":
            encoder = DNNEncoder
        elif encoder == "rnn":
            encoder = ReverseGRUEncoder
        else:
            raise ValueError("encoders only include dnn, cnn and rnn")

        self.encoders = nn.ModuleList([
            encoder(input_dim,
                    latent_dim,
                    hidden_units // 2,
                    encode_obs_time,
                    timesteps=input_timesteps) for _ in range(len(s_recon_terms))
        ])
        recon_steps = output_timesteps if pass_raw else output_timesteps + input_timesteps
        self.nlblk_list = nn.ModuleList([
            SphereSurfaceModel(
                s,
                output_dim,
                latent_dim,
                recon_steps,
                include_s_recon_terms,
                # hidden_units,
            ) for s in s_recon_terms
        ])
        NeuralLaplace.torchlaplace.inverse_laplace.device = device


        # * Shared Encoder
        # self.encoder = CNNEncoder(input_dim,
        #                           latent_dim,
        #                           hidden_units // 2,
        #                           input_timesteps,
        #                           encode_obs_time=encode_obs_time)
        # self.nlblk_list = nn.ModuleList([
        #     SphereSurfaceModel(
        #         s,
        #         output_dim,
        #         latent_dim,
        #         output_timesteps,
        #         include_s_recon_terms,
        #         hidden_units,
        #     ) for s in s_recon_terms
        # ])
        # NeuralLaplace.torchlaplace.inverse_laplace.device = device

    # TODO: amplify the magnitude of the residual?
    def forward(self, observed_data, observed_tp, tp_to_predict):
        # trajs_to_encode : (N, T, D) tensor containing the observed values.
        # tp_to_predict: Is the time to predict the values at.

        # # out = observed_data
        # forecast, recon_loss = 0, 0
        # all_tp = torch.cat([observed_tp, tp_to_predict], axis=-1)
        # for i, nlblk in enumerate(self.nlblk_list):

        #     p = self.encoder(observed_data, observed_tp)
        #     fcst = laplace_reconstruct(
        #         nlblk,
        #         p,
        #         tp_to_predict,
        #         # all_tp,
        #         ilt_reconstruction_terms=self.s_recon_terms_list[i],
        #         recon_dim=self.output_dim,
        #         use_sphere_projection=self.use_sphere_projection,
        #         ilt_algorithm=self.ilt_algorithm,
        #         options={"start_k": self.start_ks[i]})

        #     # fcst, recon = temp[:, self.
        #     #                    input_timesteps:, :], temp[:, :self.
        #     #                                               input_timesteps, :]

        #     # assert recon.shape == out.shape
        #     # # recon_loss += torch.nn.functional.mse_loss(recon, out) * 5e-1
        #     # out = out - recon
        #     forecast += fcst
        # return forecast, recon_loss

        # out = observed_data
        # forecast, recon_loss = 0, 0
        # for nlblk in self.nlblk_list:
        #     # count += 1
        #     # # avg_out = aiblk(out)
        #     # fcst = nlblk(out, observed_tp, tp_to_predict)
        #     # recon = nlblk(out, observed_tp, observed_tp)
        #     temp = nlblk(out, observed_tp,
        #                  torch.cat([observed_tp, tp_to_predict], axis=-1))
        #     fcst, recon = temp[:, self.
        #                        input_timesteps:, :], temp[:, :self.
        #                                                   input_timesteps, :]
        #     assert recon.shape == out.shape
        #     # recon_loss += torch.nn.functional.mse_loss(recon, out) * 5e-1
        #     out = out - recon
        #     forecast += fcst
        # return forecast, recon_loss

        out = observed_data
        forecast, recon_loss = 0, 0
        all_tp = torch.cat([observed_tp, tp_to_predict], axis=-1)
        for i, (encoder,
                nlblk) in enumerate(zip(self.encoders, self.nlblk_list)):
            if self.pass_raw:
                p = encoder(observed_data, observed_tp)
                fcst = laplace_reconstruct(
                    nlblk,
                    p,
                    tp_to_predict,
                    ilt_reconstruction_terms=self.s_recon_terms_list[i],
                    recon_dim=self.output_dim,
                    use_sphere_projection=self.use_sphere_projection,
                    include_s_recon_terms=self.include_s_recon_terms,
                    ilt_algorithm=self.ilt_algorithm,
                    options={"start_k": self.start_ks[i]})
                forecast += fcst
            else:
                p = encoder(out, observed_tp)
                temp = laplace_reconstruct(
                    nlblk,
                    p,
                    all_tp,
                    ilt_reconstruction_terms=self.s_recon_terms_list[i],
                    recon_dim=self.output_dim,
                    use_sphere_projection=self.use_sphere_projection,
                    include_s_recon_terms=self.include_s_recon_terms,
                    ilt_algorithm=self.ilt_algorithm,
                    options={"start_k": self.start_ks[i]})

                fcst, recon = temp[:, self.
                                   input_timesteps:, :], temp[:, :self.
                                                              input_timesteps, :]
                out = out - recon
                forecast += fcst

            # assert recon.shape == out.shape
            # # recon_loss += torch.nn.functional.mse_loss(recon, out) * 5e-1
        return forecast, recon_loss
