import argparse
import json
import logging
import os
import sys

import numpy as np
import opensfm.reconstruction as orec
from opensfm import align, dataset, log, multiview, pygeometry
from opensfm import transformations as tf
from opensfm import types

from run_ba import (
    bundle_with_fixed_images,
    compute_gcp_std,
    decompose_covariance,
    gcp_geopositional_error,
    get_sorted_reprojection_errors,
    reproject_gcps,
    triangulate_gcps,
)

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset",
        help="dataset to process",
    )
    parser.add_argument(
        "--rec",
        help="reconstruction index to use",
        type=int,
    )
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    path = args.dataset
    data = dataset.DataSet(path)
    for fn in (
        "reconstruction.json",
        "tracks.csv",
        "reference_lla.json",
        "camera_models.json",
    ):
        if not (os.path.exists(os.path.join(path, fn))):
            logger.error(f"Missing file: {fn}")
            return

    camera_models = data.load_camera_models()
    tracks_manager = data.load_tracks_manager()
    gcps = data.load_ground_control_points()

    retriangulated_reconstruction_path = data._reconstruction_file(
        "reconstruction_retriangulated.json"
    )
    if not os.path.exists(retriangulated_reconstruction_path):
        reconstructions = data.load_reconstruction()
        for reconstruction in reconstructions:
            reconstruction.add_correspondences_from_tracks_manager(tracks_manager)

        assert (
            len(reconstructions) == 1 or args.rec is not None
        ), "Expected a single reconstruction or a value for --rec"

        rec_ix = args.rec if len(reconstructions) > 1 else 0
        rec = reconstructions[rec_ix]

        # Rigidly align the reconstruction to GPS + GCP
        # Here the GCP will be mostly ignored since there are many more GPS points
        # TODO: Ignore GPS here if there are enough GCP points? Might help BA convergence
        orec.align_reconstruction(rec, gcps, data.config)

        gcp_alignment = {"after_rigid": gcp_geopositional_error(gcps, rec)}
        logger.info(
            "GCP errors after rigid alignment:\n"
            + "\n".join(
                "[{}]: {:.2f}m".format(k, v["error"])
                for k, v in gcp_alignment["after_rigid"].items()
            )
        )

        if len(gcps) > 0:
            logger.info(
                "Inflating the input standard deviation of the GPS constraints to ensure that the GCP constraints are not overwhelmed."
                "This will affect the output covariances, particularly if there are few GCPs"
            )
            for shot in rec.shots.values():
                shot.metadata.gps_accuracy.value = 0.5 * len(rec.shots)

        data.config["bundle_max_iterations"] = 200
        data.config["bundle_use_gcp"] = len(gcps) > 0

        # logger.info("Running initial BA to align")
        report = orec.bundle(rec, camera_models, {}, gcp=gcps, config=data.config)
        data.save_reconstruction([rec], f"reconstruction_ba.json")

        gcp_alignment["after_bundle"] = gcp_geopositional_error(gcps, rec)
        logger.info(
            "GCP errors after bundle :\n"
            + "\n".join(
                "[{}]: {:.2f}m".format(k, v["error"])
                for k, v in gcp_alignment["after_bundle"].items()
            )
        )
        json.dump(
            gcp_alignment,
            open(f"{data.data_path}/gcp_alignment.json", "w"),
            indent=4,
            sort_keys=True,
        )

        # Re-triangulate to remove badly conditioned points
        n_points = len(rec.points)
        logger.info("Re-triangulating...")
        backup = data.config["triangulation_min_ray_angle"]
        data.config["triangulation_min_ray_angle"] = 2.0
        orec.retriangulate(tracks_manager, rec, data.config)
        data.save_reconstruction([rec], f"reconstruction_retriangulated.json")
        data.config["triangulation_min_ray_angle"] = backup
        logger.info(
            f"Re-triangulated. Removed {n_points - len(rec.points)}."
            f" Kept {int(100*len(rec.points)/n_points)}%"
        )
    else:
        logger.info("Skipping to second BA. Loading re-triangulated reconstruction")
        reconstructions = data.load_reconstruction("reconstruction_retriangulated.json")
        for reconstruction in reconstructions:
            reconstruction.add_correspondences_from_tracks_manager(tracks_manager)
        assert (
            len(reconstructions) == 1
        ), "Expected a single reconstruction in the retriangulated file"
        rec = reconstructions[0]

    # Reproject GCPs with a very loose threshold so that we get a point every time
    # We use this for annotator feedback and to compute the covariance of the reprojections
    if len(gcps) > 0:
        gcp_reprojections = reproject_gcps(gcps, rec, reproj_threshold=10)
        reprojection_errors = get_sorted_reprojection_errors(gcp_reprojections)
        err_values = [t[2] for t in reprojection_errors]
        max_reprojection_error = np.max(err_values)
        median_reprojection_error = np.median(err_values)

        json.dump(
            gcp_reprojections,
            open(
                f"{data.data_path}/gcp_reprojections.json",
                "w",
            ),
            indent=4,
            sort_keys=True,
        )

        gcp_std = compute_gcp_std(gcp_reprojections)
        logger.info(f"GCP reprojection error STD: {gcp_std}")
    else:
        gcp_std = None

    # Run the second BA, this time using the computed reprojection std. deviation for the annotated GCPs
    logger.info("Running second BA...")
    covariance_estimation_valid = bundle_with_fixed_images(
        rec,
        camera_models,
        gcp=gcps,
        gcp_std=gcp_std,
        fixed_images=(),
        config=data.config,
    )
    logger.info("Done running second BA")
    data.save_reconstruction([rec], f"reconstruction_ba_2.json")
    if not covariance_estimation_valid:
        logger.info(
            f"Could not get positional uncertainty. It could be because:"
            "\na) there are not enough GCPs."
            "\nb) they are badly distributed in 3D."
            "\nc) there are some wrong annotations"
        )
        shots_std = [(shot, np.nan) for shot in rec.shots]
    else:
        shots_std = []
        for shot in rec.shots.values():
            u, std_v = decompose_covariance(shot.covariance[3:, 3:])
            std = np.linalg.norm(std_v)
            shots_std.append((shot.id, std))

    std_values = [x[1] for x in shots_std]
    n_nan_std = sum(np.isnan(std) for std in std_values)
    n_zero_std = sum(np.isclose(std, 0) for std in std_values)

    # Average positional STD
    median_shot_std = np.median(std_values)

    # Save the shot STD to a file
    with open(f"{data.data_path}/shots_std.csv", "w") as f:
        s = sorted(shots_std, key=lambda t: -t[-1])
        for t in s:
            line = "{}, {}".format(*t)
            f.write(line + "\n")

        max_shot_std = s[0][1]


if __name__ == "__main__":
    log.setup()
    sys.exit(main())
