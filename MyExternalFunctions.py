## importing libraries
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors


def plot_settings(style_name='seaborn-dark', grid=False, axes_ls=14, legend_fs=14):
    # Changing plotting style
    import matplotlib as mpl
    plt.style.use(style_name)
    plot_format = {
        'axes.grid': grid,
        'axes.labelsize': axes_ls,
        'legend.fontsize': legend_fs,
        'font.family': ['serif'],
        'xtick.labelsize': legend_fs,
        'ytick.labelsize': legend_fs,
        'mathtext.fontset': 'dejavuserif',
        'mathtext.rm'  : 'serif',
        'mathtext.it'  : 'serif:italic',
        'mathtext.bf'  : 'serif:bold',
    }
    for p in plot_format:
        mpl.rcParams[p] = plot_format[p]


########################################## PD strategy update function #################################################
def _update_PD_game(S, b,show_defector_switch=False, show_cooperator_switch=False):
    '''
    params:
        S: 
            The current strategy lattice.
                0: cooperator
                1: defector
        b:  
            Advantage for cheating
        show_defector_switch:
            Determines if the players that switch strategy from cooperator to defector is shown (default is False)  
        show_cooperator_switch
            Determines if the players that switch strategy from defector to cooperator is shown (default is False)  
    returns:
        S: The updated strategy lattice.
        C: The color values used for plotting:
                C[0,0]: cooperator (no switch)
                C[0,1]: cooperator (switch)
                C[1,0]: defector (no switch)
                C[1,1]: defector (switch)
    '''
    color = np.array(([0, 6],[11, 16]))

    N = S.shape[0]
    S_new = np.zeros(S.shape, dtype=int) # new strategy matrix
    pm = np.array(([1, 0],[b, 0]))       # payoff matrix
    payoff = np.zeros(S.shape)           # payout matrix
    C = np.zeros((N,N))                  # the matrix that stores the color values used for plotting

    bc = np.zeros(2 * N + 1, dtype=int)
    for i in np.arange(0, N):  # setting up boundaries
        bc[i] = i
    bc[-1] = N-1; bc[N] = 0;  bc = bc.astype(int)


    for i in np.arange(0,N, dtype=int):
        for j in np.arange(0,N, dtype=int):
            pa = 0
            for k in np.arange(-1,2, dtype=int):
                for l in np.arange(-1,2, dtype=int):
                    # if k == 0 and l == 0:
                    #     pass
                    # else:
                    pa = pa + pm[S[i,j],S[bc[i+k],bc[j+l]]]
            payoff[i,j] = pa

    for i in np.arange(0,N, dtype=int):
        for j in np.arange(0, N, dtype=int):
            hp = payoff[i,j]
            S_new[i,j] = S[i,j]
            for k in np.arange(-1, 2, dtype=int):
                for l in np.arange(-1, 2, dtype=int):
                    if payoff[bc[i+k],bc[j+l]] > hp:
                        hp = payoff[bc[i+k],bc[j+l]]
                        S_new[i,j] = S[bc[i+k],bc[j+l]]

    for i in np.arange(0,N, dtype=int):
        for j in np.arange(0, N, dtype=int):
            ci, cj = [S_new[i,j]]*2 
            if show_defector_switch:
                if S[i,j] > S_new[i,j]: 
                    ci = S[i,j]
            if show_cooperator_switch:
                if S[i,j] < S_new[i,j]: 
                    ci = S[i,j]
            C[i,j] = color[ci,cj]
            S[i,j] = S_new[i,j]

            # if mode == 1:
            #     C[i,j] = color[S[i,j],S_new[i,j]]
            #     S[i,j] = S_new[i,j]
            # else:
            #     C[i, j] = color[S[i, j], S[i, j]]
                # S[i, j] = S_new[i, j]

    return C, S


########################################## PD run and plot function #################################################
def run_PD_game(fn, S, b, show_defector_switch=False, show_cooperator_switch=False):
    from matplotlib.animation import FuncAnimation
    from MyExternalFunctions import plot_settings
    plot_settings()
    
    c1 = '#3f3f7f'                  # Blue (cooperation)
    c2 = '#bf3f00'                  # Red (defector)
    c3 = '#ffdf00'                  # Yellow (change to defector)
    c4 = '#3f7f3f'                  # Green (change to cooperation)
    cmap = colors.ListedColormap([c1, c3, c4, c2])
    bounds=[0,5,10,15,20]
    norm = colors.BoundaryNorm(bounds, cmap.N)
    color = np.array(([0, 6],[11, 16])) 
    
    C0 = color[S,S]
    PD_params = dict(S=S, b=b, show_defector_switch=show_defector_switch, show_cooperator_switch=show_cooperator_switch)
    
    x = np.arange(fn)
    y = np.zeros_like(x, dtype=float)
    fig, (ax1, ax2) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, 2]}, figsize=(16,5))
    im = ax1.imshow(C0, animated=True, cmap=cmap, vmin=0, vmax=16)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.grid(False)
    
    l2, = ax2.plot([],[], color=c1, lw=2, label='Cooperators')
    l1, = ax2.plot([],[], color=c2, lw=2, label='Defector')
    ax2.set_xlim([-1, fn+1])
    ax2.set_xlabel('Itterations')
    ax2.set_ylim([0, 1])
    ax2.set_ylabel('Fraction')
    ax2.legend(ncol=2, bbox_to_anchor=(1,1.15))
    ax2.grid(True)
    

    def init():
        im.set_data(C0)
        y[0] = np.count_nonzero(C0)/np.prod(C0.shape)
        l1.set_data(x[0], y[0])
        l2.set_data(x[0], 1-y[0])
        return [im, l1, l2]


    def animate(i, PD_params):
        if i == 0:
            im.set_data(C0)
            y[0] = np.count_nonzero(C0)/np.prod(C0.shape)
            l1.set_data(x[0], y[0])
            l2.set_data(x[0], 1-y[0])
        else:
            C, S = _update_PD_game(**PD_params)
            PD_params['S'] = S
            im.set_data(C)

            y[i] = np.count_nonzero(C)/np.prod(C.shape)
            l1.set_data(x[:i], y[:i])
            l2.set_data(x[:i], 1-y[:i])


        return [im, l1, l2]

    ani = FuncAnimation(fig, animate, fargs=[PD_params], frames=fn, interval=20, init_func=init)
    # plt.tight_layout()
    plt.close()
    
    return ani