import os
import argparse
import core.logger as logger

import results.ensemble as ens
from bronx.stdtypes.date import daterangex as rangex

# import warnings
# warnings.filterwarnings("ignore")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/sr_example.jsonc',
                        help='JSON file for configuration')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)

    # parse configs
    args = parser.parse_args()
    opt = logger.parse(args)

    # create dirs
    for _, item in opt["path"].items():
        os.makedirs(item, exist_ok=True)

    ens_opt_arome = opt["ensemble"]["Arome"]
    ens_opt_ddpm  = opt["ensemble"]["DDPM"]

    # load ensembles
    arome_ensemble = ens.load_ensemble_arome(
        dates=ens_opt_arome["dates"],
        echeances=ens_opt_arome["echeances"],
        params=opt["data_loading"]["params_out"],
        n_members=ens_opt_arome["n_members"],
        data_location=ens_opt_arome["data_location"]
    )
    arome_ensemble = ens.correct_dates_for_arome(arome_ensemble)

    ddpm_ensemble = ens.load_ensemble(
        working_dir=opt["path"]["working_dir"],
        n_members=ens_opt_ddpm["n_members"],
        params=opt["data_loadiing"]["params_out"]
    )

    for i_p, p in enumerate(opt["params_out"]):

        output_dir = os.path.join(opt["path"]["working_dir"], p)
        os.makedirs(output_dir)
        output_dir_arome = os.path.join(output_dir, "arome")
        os.makedirs(output_dir_arome)
        output_dir_ddpm = os.path.join(output_dir, "ddpm")
        os.makedirs(output_dir_ddpm)
        

        # plot maps for all ensembles
        ens.plot_maps_ensemble(
            arome_ensemble,
            output_dir_arome,
            param=p,
            n_members=ens_opt_arome["n_members"],
            n=opt["results"]["images"]["n"],
            cmap=opt["results"]["images"]["cmap"][i_p]
        )

        ens.plot_maps_ensemble(
            ddpm_ensemble,
            output_dir_ddpm,
            param=p,
            n_members=ens_opt_ddpm["n_members"],
            n=opt["results"]["images"]["n"],
            cmap=opt["results"]["images"]["cmap"][i_p]
        )

        # plot stats
        mean_arome = ens.compute_pointwise_mean(arome_ensemble, ens_opt_arome["n_members"], opt["data_loading"]["params_out"])
        std_arome  = ens.compute_pointwise_std(arome_ensemble, ens_opt_arome["n_members"], opt["data_loading"]["params_out"])
        Q5_arome   = ens.compute_pointwise_Q5(arome_ensemble, ens_opt_arome["n_members"], opt["data_loading"]["params_out"])
        Q95_arome  = ens.compute_pointwise_Q95(arome_ensemble, ens_opt_arome["n_members"], opt["data_loading"]["params_out"])
        mean_ddpm  = ens.compute_pointwise_mean(ddpm_ensemble , ens_opt_ddpm["n_members"] , opt["data_loading"]["params_out"])
        std_ddpm   = ens.compute_pointwise_std(ddpm_ensemble , ens_opt_ddpm["n_members"] , opt["data_loading"]["params_out"])
        Q5_ddpm    = ens.compute_pointwise_Q5(ddpm_ensemble , ens_opt_ddpm["n_members"] , opt["data_loading"]["params_out"])
        Q95_ddpm   = ens.compute_pointwise_Q95(ddpm_ensemble , ens_opt_ddpm["n_members"] , opt["data_loading"]["params_out"])

        mean = ens.group_ensembles(mean_arome, mean_ddpm) 
        std  = ens.group_ensembles(std_arome, std_ddpm) 
        Q95  = ens.group_ensembles(Q5_arome, Q5_ddpm)
        Q5   = ens.group_ensembles(Q95_arome, Q95_ddpm)
    
        ens.synthesis_unique_all_stats_ensemble(
            mean,
            std,
            Q5,
            Q95,
            output_dir,
            p
        )

        ens.synthesis_stat_distrib(
            mean,
            std,
            Q5,
            Q95,
            output_dir,
            p
        )
