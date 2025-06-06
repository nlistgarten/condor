from dataclasses import asdict

import matplotlib.pyplot as plt


def LTI_plot(sim, t_slice=None):
    if t_slice is None:
        t_slice = slice(None, None)
    field = sim.state
    # for field in [sim.state, sim.output]:
    for sym_name, symbol in asdict(field).items():
        n = symbol.shape if symbol.ndim > 1 else 1
        fig, axes = plt.subplots(n, 1, constrained_layout=True, sharex=True)
        plt.suptitle(f"{sim.__class__.__name__} {field.__class__.__name__}.{sym_name}")
        if n > 1:
            for ax, x in zip(axes, symbol):
                ax.plot(sim.t[t_slice], x.squeeze()[t_slice])
                ax.grid(True)
        else:
            plt.plot(sim.t[t_slice], symbol.squeeze()[t_slice])
            plt.grid(True)
