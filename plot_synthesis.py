import os
import argparse
import json
from collections import OrderedDict
import results.outputs as outputs
import results.pointwise_scores as ps
import results.WD as wd
import results.PSD as psd
import results.correlation_length as corr_len
import results.correlation as corr
import warnings
warnings.filterwarnings("ignore")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/sr_example.jsonc',
                        help='JSON file for configuration')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)

    # parse configs
    args = parser.parse_args()
    opt_path = args.config
    # remove comments starting with '//'
    json_str = ''
    with open(opt_path, 'r') as f:
        for line in f:
            line = line.split('//')[0] + '\n'
            json_str += line
    opt = json.loads(json_str, object_pairs_hook=OrderedDict)

    # create dirs
    for _, item in opt["path"].items():
        os.makedirs(item, exist_ok=True)

    expes_names = opt["expes"]["name"]
    expes_paths = opt["expes"]["results"]
    expes_interps = opt["expes"]["interp"]

    print("Loading results ...")
    expes_results = [
        outputs.load_results(
            expes_paths[i], 
            expes_interps[i], 
            opt["data"]["data_test_location"], 
            opt["data"]["baseline_location"], 
            param=opt["data"]["param"]
        ) for i in range(len(expes_paths))
    ]

    # plot maps
    print("Ploting maps ...")
    outputs.plot_synthesis_maps(
        expes_names,
        expes_results,
        opt["path"]["output_dir"],
        opt["data"]["param"],
        opt["data"]["unit"], 
        cmap=opt["maps"]["cmap"],
        several_inputs=opt["expes"]["several_inputs"]
    )

    # pointwise scores
    if opt["mae"]["maps"]:
        print("Computing & Plotting MAE maps ...")
        maes_df = []
        for expe_df in expes_results:
            maes_df.append(ps.compute_score(expe_df, ps.mae, "MAE"))
        ps.plot_synthesis_scores(
            expes_names,
            maes_df,
            opt["path"]["output_dir"],
            "MAE",
            opt["data"]["unit"], 
            cmap="pink"
        )
    if opt["mae"]["distribs"]:
        print("Computing & Plotting MAE distributions ...")
        maes_df = []
        maes_df_terre = []
        maes_df_mer = []
        for expe_df in expes_results:
            maes_df.append(ps.compute_score(expe_df, ps.mae, "MAE"))
            maes_df_terre.append(ps.compute_score_terre(maes_df[-1], "MAE"))
            maes_df_mer.append(ps.compute_score_mer(maes_df[-1], "MAE"))
        ps.synthesis_score_distribs(
            expes_names,
            maes_df,
            maes_df_terre,
            maes_df_mer,
            opt["path"]["output_dir"],
            "MAE"
        )


    if opt["mse"]["maps"]:
        print("Computing & Plotting MSE maps ...")
        mses_df = []
        for expe_df in expes_results:
            mses_df.append(ps.compute_score(expe_df, ps.mse, "MSE"))
        ps.plot_synthesis_scores(
            expes_names,
            mses_df,
            opt["path"]["output_dir"],
            "MSE",
            "$" + opt["data"]["unit"] + "^2$", 
            cmap="pink"
        )
    if opt["mse"]["distribs"]:
        print("Computing & Plotting MSE distributions ...")
        mses_df = []
        mses_df_terre = []
        mses_df_mer = []
        for expe_df in expes_results:
            mses_df.append(ps.compute_score(expe_df, ps.mse, "MSE"))
            mses_df_terre.append(ps.compute_score_terre(mses_df[-1], "MSE"))
            mses_df_mer.append(ps.compute_score_mer(mses_df[-1], "MSE"))
        ps.synthesis_score_distribs(
            expes_names,
            mses_df,
            mses_df_terre,
            mses_df_mer,
            opt["path"]["output_dir"],
            "MSE"
        )


    if opt["bias"]["maps"]:
        print("Computing & Plotting bias maps ...")
        bias_df = []
        for expe_df in expes_results:
            bias_df.append(ps.compute_score(expe_df, ps.bias, "bias"))
        ps.plot_synthesis_scores(
            expes_names,
            bias_df,
            opt["path"]["output_dir"],
            "bias",
            opt["data"]["unit"], 
            cmap="coolwarm"
        )
    if opt["bias"]["distribs"]:
        print("Computing & Plotting bias distributions ...")
        bias_df = []
        bias_df_terre = []
        bias_df_mer = []
        for expe_df in expes_results:
            bias_df.append(ps.compute_score(expe_df, ps.bias, "bias"))
            bias_df_terre.append(ps.compute_score_terre(bias_df[-1], "bias"))
            bias_df_mer.append(ps.compute_score_mer(bias_df[-1], "bias"))
        ps.synthesis_score_distribs(
            expes_names,
            bias_df,
            bias_df_terre,
            bias_df_mer,
            opt["path"]["output_dir"],
            "bias"
        )
        

    if opt["ssim"]["maps"]:
        print("Computing & Plotting SSIM maps ...")
        ssim_df = []
        for expe_df in expes_results:
            ssim_df.append(ps.compute_score(expe_df, ps.ssim, "SSIM"))
        ps.plot_synthesis_scores(
            expes_names,
            ssim_df,
            opt["path"]["output_dir"],
            "SSIM",
            "", 
            cmap="plasma"
        )
    if opt["ssim"]["distribs"]:
        print("Computing & Plotting SSIM distributions ...")
        ssim_df = []
        ssim_df_terre = []
        ssim_df_mer = []
        for expe_df in expes_results:
            ssim_df.append(ps.compute_score(expe_df, ps.ssim, "SSIM"))
            ssim_df_terre.append(ps.compute_score_terre(ssim_df[-1], "SSIM"))
            ssim_df_mer.append(ps.compute_score_mer(ssim_df[-1], "SSIM"))
        ps.synthesis_score_distribs(
            expes_names,
            ssim_df,
            ssim_df_terre,
            ssim_df_mer,
            opt["path"]["output_dir"],
            "SSIM"
        )
        

    
    # WD
    if opt["WD"]:
        print("Computing & Plotting WD ...")
        wds_df = []
        wds_df_terre = []
        wds_df_mer = []
        for expe_df in expes_results:
            wds_df.append(wd.compute_datewise_WD(expe_df))
            wds_df_terre.append(wd.compute_datewise_WD_terre(expe_df))
            wds_df_mer.append(wd.compute_datewise_WD_mer(expe_df))
        wd.synthesis_wasserstein_distance_distrib(
            expes_names,
            wds_df,
            wds_df_terre,
            wds_df_mer,
            opt["path"]["output_dir"]
        )


    # PSDs
    if opt["PSD"]:
        print("Computing & Plotting PSDs ...")
        psds_df = []
        for expe_df in expes_results:
            psds_df.append(psd.PSD(expe_df))
        psd.synthesis_PSDs(expes_names, psds_df, opt["path"]["output_dir"], several_inputs=opt["expes"]["several_inputs"])


    # correlation length
    if opt["corr_len"]:
        print("Computing & Plotting correlation lengths ...")
        corr_lens_df = []
        for expe_df in expes_results:
            corr_lens_df.append(corr_len.compute_corr_len(expe_df))
        corr_len.plot_synthesis_corr_len(
            expes_names,
            corr_lens_df,
            opt["path"]["output_dir"]
        )

    # correlation
    if opt["corr"]:
        print("Computing & Plotting correlations ...")
        corrs_df = []
        corrs_df_terre = []
        corre_df_mer = []
        for expe_df in expes_results:
            corrs_df.append(corr.correlation(expe_df))
            corrs_df.append(corr.correlation_terre(expe_df))
            corrs_df.append(corr.correlation_mer(expe_df))