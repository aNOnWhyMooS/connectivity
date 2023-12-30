from typing import Dict, Tuple
import torch
import torch.nn as nn
import numpy as np
import math
from .utils import flatten, unflatten_like
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import math


def get_basis(model, anchor=0, base1=1, base2=2):
    n_vert = model.n_vert
    n_par = int(sum([p.numel() for p in model.parameters()]) / n_vert)

    if n_vert <= 2:
        return torch.randn(n_par, 1), torch.randn(n_par, 1)
    else:
        par_vecs = torch.zeros(n_vert, n_par)
        if torch.has_cuda:
            par_vecs = par_vecs.cuda()
        for ii in range(n_vert):
            temp_pars = [p for p in model.net.parameters()][ii::n_vert]
            par_vecs[ii, :] = flatten(temp_pars)

        first_pars = torch.cat((n_vert * [par_vecs[anchor, :].unsqueeze(0)]))
        diffs = par_vecs - first_pars
        dir1 = diffs[base1, :]
        dir2 = diffs[base2, :]

        ## now gram schmidt these guys ##
        vu = dir2.squeeze().dot(dir1.squeeze())
        uu = dir1.squeeze().dot(dir1.squeeze())

        dir2 = dir2 - dir1.mul(vu).div(uu)

        ## normalize ##
        dir1 = dir1.div(dir1.norm())
        dir2 = dir2.div(dir2.norm())

        return dir1.unsqueeze(-1), dir2.unsqueeze(-1)


def compute_loss_surface(model, train_x, train_y, v1, v2, loss, n_pts=50, range_=10.0):
    start_pars = model.state_dict()
    vec_len = torch.linspace(-range_.item(), range_.item(), n_pts)
    ## init loss surface and the vector multipliers ##
    loss_surf = torch.zeros(n_pts, n_pts)
    with torch.no_grad():
        ## loop and get loss at each point ##
        for ii in range(n_pts):
            for jj in range(n_pts):
                perturb = v1.mul(vec_len[ii]) + v2.mul(vec_len[jj])
                # print(perturb.shape)
                perturb = unflatten_like(perturb.t(), model.parameters())
                for i, par in enumerate(model.parameters()):
                    par.data = par.data + perturb[i].to(par.device)

                loss_surf[ii, jj] = loss(model(train_x), train_y)

                model.load_state_dict(start_pars)

    X, Y = np.meshgrid(vec_len, vec_len)
    return X, Y, loss_surf


def compute_loader_loss(
    model, loader, loss, n_batch, binary=False, device=torch.device("cuda:0")
):
    total_loss = torch.tensor([0.0])
    for i, input in enumerate(loader):
        if i < n_batch:
            input_ids = input["input_ids"][0].cuda()
            attention_mask = input["attention_mask"][0].cuda()
            if "token_type_ids" in input.keys():
                type_ids = input["token_type_ids"][0].cuda()
            else:
                type_ids = None
            if "label" in input.keys():
                target = input["label"].cuda()

            output = model(input_ids, attention_mask, type_ids, target)
            loss_ = output.loss.item()
            if binary:
                true = input["label"].cuda()
                with torch.no_grad():
                    logits = output.logits.cpu()
                    x = torch.max(
                        torch.concat(
                            (logits[:, 0].view(-1, 1), logits[:, 1].view(-1, 1)), 1
                        ),
                        1,
                    ).values
                    logits = torch.concat((logits[:, 1].view(-1, 1), x.view(-1, 1)), 1)
                    loss_ = loss(logits.cuda(), true).item()

            total_loss += loss_

        else:
            break

    return total_loss


def compute_loss_surface_loader(
    model,
    loader,
    v1,
    v2,
    binary=False,
    loss=torch.nn.CrossEntropyLoss(),
    n_batch=10,
    n_pts=50,
    range_=10.0,
    device=torch.device("cuda:0"),
):
    start_pars = model.state_dict()
    vec_len = torch.linspace(-range_.item(), range_.item(), n_pts)
    ## init loss surface and the vector multipliers ##
    loss_surf = torch.zeros(n_pts, n_pts)
    with torch.no_grad():
        ## loop and get loss at each point ##
        for ii in range(n_pts):
            for jj in range(n_pts):
                perturb = v1.mul(vec_len[ii]) + v2.mul(vec_len[jj])
                # print(perturb.shape)
                perturb = unflatten_like(perturb.t(), model.parameters())
                for i, par in enumerate(model.parameters()):
                    par.data = par.data + perturb[i].to(par.device)

                loss_surf[ii, jj] = compute_loader_loss(
                    model, loader, loss, n_batch, device=device
                )

                model.load_state_dict(start_pars)

    X, Y = np.meshgrid(vec_len, vec_len)
    return X, Y, loss_surf


def plot_loss_surface(
    loss_surface, savename, three_d=True, locations: Dict[str, Tuple[float, float]] = {}
):
    """
    Args:
        locations:  Names and locations of points to annotate on the graph.
                    For 2-D plots only, currently.
    """
    xx, yy, f = loss_surface
    xmin, xmax = min(xx.reshape(-1)), max(xx.reshape(-1))
    ymin, ymax = min(yy.reshape(-1)), max(yy.reshape(-1))
    f = np.transpose(np.array(f))

    if three_d:
        fig = plt.figure(figsize=(15, 12))
        ax = fig.add_subplot(111, projection="3d")
        ax.grid(False)
        ax.set_axis_off()

        # Plot the surface.
        surf = ax.plot_surface(
            xx, yy, f, cmap=cm.inferno, linewidth=0, antialiased=False
        )

        # Customize the z axis.
        zlims = (math.floor(np.min(f)), math.ceil(np.max(f)))
        print(zlims)
        ax.set_zlim(zlims)
        ax.zaxis.set_major_locator(LinearLocator(10))
        # A StrMethodFormatter is used automatically
        # ax.zaxis.set_major_formatter('{x:.02f}')
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])
        ax.zaxis.set_ticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

        # Add a color bar which maps values to colors.
        # fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.savefig(savename + ".pdf")  # , transparency=True)
        plt.savefig(savename + ".png")  # , transparency=True)

    else:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.gca()
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        cfset = ax.contourf(xx, yy, f, levels=40, cmap="coolwarm")
        ax.imshow(np.transpose(f), cmap="coolwarm", extent=[xmin, xmax, ymin, ymax])
        plt.colorbar(cfset)
        # cset = ax.contour(xx, yy, f, colors='k')
        # ax.clabel(cset, inline=1, fontsize=10)
        # ax.set_xticks([])
        # ax.set_yticks([])

        points_x = []
        points_y = []
        for name, location in locations.items():
            plt.annotate(name, xy=location)
            points_x.append(location[0])
            points_y.append(location[1])
        points_x.append(points_x[0])
        points_y.append(points_y[0])
        plt.plot(points_x, points_y, linestyle="dashed", marker="s", color="k")

        plt.savefig(
            savename + ".pdf",
        )  # transparency=True)
        plt.savefig(
            savename + ".png",
        )  # transparency=True)
    # plt.title('Loss Surface')

    # plt.show()
