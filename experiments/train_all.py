"""Convenience launcher for sequentially running the comparison baselines."""

import _bootstrap  # noqa: F401
from pid_saca.algorithms import saca as SACA
from pid_saca.algorithms import grc_elg as GRC_ELG
from pid_saca.algorithms import saca_baseline as SACA_baseline
from pid_saca.algorithms import jdrl_pomo as JDRL_POMO
import matplotlib.pyplot as plt

def main(A=10, w=1):

    plt.show = lambda: None

    SACA_config = SACA.Config()
    GRC_ELG_config = GRC_ELG.Config()
    SACA_baseline_config = SACA_baseline.Config()
    JDRL_POMO_config = JDRL_POMO.Config()

    SACA_config.demand_fluctuation, SACA_config.demand_frequency = A, w
    GRC_ELG_config.demand_fluctuation, GRC_ELG_config.demand_frequency = A, w
    SACA_baseline_config.demand_fluctuation, SACA_baseline_config.demand_frequency = A, w
    JDRL_POMO_config.demand_fluctuation, JDRL_POMO_config.demand_frequency = A, w


    JDRL_POMO.train_jdrl(JDRL_POMO_config)


    SACA.train_saca(SACA_config)


    GRC_ELG.train_grc_elg(GRC_ELG_config)


    SACA_baseline.train_saca(SACA_baseline_config)

if __name__ == "__main__":
    main()
