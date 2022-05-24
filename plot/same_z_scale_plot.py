import argparse
import pickle
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
plt.style.use("../paper.mplstyle")

def get_parser():
    parser = argparse.ArgumentParser(description='Plot multiple surfaces, with same Z-axis scale.')
    parser.add_argument("--surface_pkl_files", nargs='+', type=str, required=True, 
                        help="All the pkl files whose surfaces are to be plotted with same Z-scale.")
    parser.add_argument("--plot_title", type=str, default=None, help="Title of the plot.")
    parser.add_argument("--names", nargs="+", type=str, default=None, help="Names of the surfaces.")
    parser.add_argument("--savename", type=str, default="temp", help="File where the final plot is to be saved.")
    parser.add_argument("--clip_x", type=float, nargs=2, help="Will clip graph to be within these x values.")
    parser.add_argument("--clip_y", type=float, nargs=2, help="Will clip graph to be within these y values.")
    parser.add_argument("--clip_z", type=float, nargs=2, help="Will clip graph to be within these z values.")
    parser.add_argument("--points_per_level", type=int, default=20, help="How many points to put in same level?")
    parser.add_argument("--point_names", nargs="+", help="Names for points in order of surface_pkl_files")
    return parser

def plot_loss_surface(surfaces, savename, points_per_level, surface_names=None, 
                      plot_title=None, clip_x1=None, clip_x2=None, clip_y1=None,
                      clip_y2=None, clip_z1=None, clip_z2=None, point_names=None):
    
    def preprocess(surface):
        xx, yy, f = surface
        if clip_z1 is not None:
            f[f<clip_z1] = clip_z1
        if clip_z2 is not None:
            f[f>clip_z2] = clip_z2
        f = np.transpose(np.array(f))
        xmin, xmax = min(xx.reshape(-1)), max(xx.reshape(-1))
        ymin, ymax = min(yy.reshape(-1)), max(yy.reshape(-1))
        return (xmin, xmax), (ymin, ymax), xx, yy, f
    
    def plot_locations(ax, locations, point_name_loc):
        points_x = []
        points_y = []
        print("Actual Locations(in order of seed in .pkl file name):", locations)
        print("Renamed Locations:", point_names[point_name_loc:point_name_loc+len(locations)])
        for name, location in locations.items():
            if point_names is not None:
                name = point_names[point_name_loc]
                point_name_loc+=1
            ax.annotate(rf"${name[0]}_{name[1]}$", xy=location, fontsize=30)
            points_x.append(location[0])
            points_y.append(location[1])
        points_x.append(points_x[0])
        points_y.append(points_y[0])
        ax.plot(points_x, points_y, linestyle="dashed", marker="s", color="k")
        return ax, point_name_loc
    
    fig, axs = plt.subplots(nrows=1, ncols=len(surfaces), figsize=(8*len(surfaces), 8))
    
    xlimss, ylimss = [], []
    for (surface, locations) in surfaces:
        xlims, ylims, xx, yy, f = preprocess(surface)
        xlimss.append(xlims)
        ylimss.append(ylims)
    
    xlims = (max([e[0] for e in xlimss]+([clip_x1] if clip_x1 is not None else [])), 
             min([e[1] for e in xlimss]+([clip_x2] if clip_x2 is not None else [])))
    ylims = (max([e[0] for e in ylimss]+([clip_y1] if clip_y1 is not None else [])), 
             min([e[1] for e in ylimss]+([clip_y2] if clip_y2 is not None else [])))
    
    print("x and y limits:", xlims, ylims)
    
    vmin, vmax = 100, -100
    tot_points = 0
    for (surface, locations) in surfaces:
        xx, yy, f = surface
        index = np.logical_and(np.logical_and(xx>=xlims[0], xx<=xlims[1]), 
                               np.logical_and(yy>=ylims[0], yy<=ylims[1]))
        f = np.transpose(f.numpy())
        
        if clip_z1 is not None:
            f[f<clip_z1] = clip_z1
        if clip_z2 is not None:
            f[f>clip_z2] = clip_z2
        
        selected = f[index]
        tot_points += selected.size
        print(f"Selecting {selected.size} points to plot from {index.size} points.")
        if np.min(selected)<vmin:
            vmin = np.min(selected)
        if np.max(selected)>vmax:
            vmax = np.max(selected)

    print("Color range:", vmin, vmax)
    levels = np.linspace(vmin, vmax, tot_points//(len(surfaces)*points_per_level))

    point_name_loc=0
    for i, ((surface, locations), ax) in enumerate(zip(surfaces, axs)):
        _, _, xx, yy, f = preprocess(surface)
        ax.set_xlim(*xlims)
        ax.set_ylim(*ylims)
        
        cs = ax.contourf(xx, yy, f, cmap='coolwarm', levels=levels)    
        
        ax, point_name_loc = plot_locations(ax, locations, point_name_loc)
        
        if surface_names is not None:
            ax.set_xlabel(surface_names[i])
        
        ax.imshow(
            f, cmap='coolwarm', 
            extent=list(xlims+ylims)
        )
    
    if plot_title is not None:
        fig.suptitle(plot_title, fontsize=30)
    
    plt.colorbar(cs, ax=axs,)
    
    plt.savefig(savename + '.pdf', )
    plt.savefig(savename + '.png', )
    

def main(args):
    surfaces = []
    for filename in args.surface_pkl_files:
        with open(filename, "rb") as f:
            try:
                surface, locations = pickle.load(f)
                print("Loaded", filename, "succesfully.")
            except:
                try:
                    surface = pickle.load(f)
                    locations = {}
                    print("Loaded", filename, "with no locations to mark.")
                except BaseException as e:
                    print("Error loading file:", filename)
                    continue
            surfaces.append((surface, locations))
    
    with mpl.rc_context({'axes.titlesize' : 30, 'axes.labelsize' : 25, 'lines.linewidth' : 2, 
                         'lines.markersize' : 10, 'xtick.labelsize' : 21, 'ytick.labelsize' : 21, 
                         'figure.autolayout': False, 'legend.fontsize': 24}):
        
        plot_loss_surface(surfaces, args.savename, args.points_per_level, args.names, args.plot_title,
                        clip_x1 = (args.clip_x[0] if args.clip_x is not None else None),
                        clip_x2 = (args.clip_x[1] if args.clip_x is not None else None),
                        clip_y1 = (args.clip_y[0] if args.clip_y is not None else None),
                        clip_y2 = (args.clip_y[1] if args.clip_y is not None else None),
                        clip_z1 = (args.clip_z[0] if args.clip_z is not None else None),
                        clip_z2 = (args.clip_z[1] if args.clip_z is not None else None),
                        point_names=args.point_names)


if __name__=="__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)