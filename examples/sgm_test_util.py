import matplotlib.pyplot as plt
from dataclasses import asdict

def LTI_plot(sim,t_slice=slice(None,None)):
    field = sim.state
    #for field in [sim.state, sim.output]:
    for sym_name, symbol in asdict(field).items():
        if symbol.ndim > 1:
            n = symbol.shape[0]
        else:
            n = 1
        fig, axes = plt.subplots(n,1, constrained_layout=True, sharex=True)
        plt.suptitle(f"{sim.__class__.__name__} {field.__class__.__name__}.{sym_name}")
        if n > 1:
            for ax, x in zip(axes, symbol):
                ax.plot(sim.t[t_slice], x.squeeze()[t_slice])
                ax.grid(True)
        else:
            plt.plot(sim.t[t_slice], symbol.squeeze()[t_slice])
            plt.grid(True)

