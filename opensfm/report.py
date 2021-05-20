import logging
import os
import subprocess
from urllib.parse import quote

try:
    import pdfkit
except ImportError:
    pdfkit = None

import PIL

from opensfm import io


logger = logging.getLogger(__name__)


css_style = """
body {
    font-family: Helvetica, Arial, sans-serif;
    color: rgb(99, 115, 129);
}
h1, h2, h2 {
    margin-top: 40px;
}
h1 {
    color: rgb(5, 180, 80);
    text-align: center;
}
#subtitle {
    text-align: right;
    font-size: small;
}
.centered-image {
    text-align: center;
    margin-bottom: 20px;
}
table {
    width: 100%;
}
table, th, td {
    border: 1px solid lightgray;
    border-collapse: collapse;
    margin-bottom: 20px
}
th, td {
    padding: 7px;
    text-align: left;
}
tr:nth-child(even) {
    background-color: #eee;
}
tr:nth-child(odd) {
    background-color: #fff;
}
th {
    background-color: #dfe;
}
.row-header td:first-child {
    background-color: #dfe;
}
.block {
    margin-top: 20px;
    page-break-inside: avoid;
}
"""


def _git_version():
    out, _ = subprocess.Popen(
        ["git", "describe", "--tags"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).communicate()
    return out.strip().decode()


class Report:
    def __init__(self, data):
        self.output_path = os.path.join(data.data_path, "stats")
        self.dataset_name = os.path.basename(data.data_path)

        self.stats = self._read_stats_file("stats.json")

    def save_report(self, filename):
        html_str = self.html
        html_path = os.path.join(self.output_path, f"{filename}.html")
        pdf_path = os.path.join(self.output_path, f"{filename}.pdf")
        with io.open_wt(html_path) as fp:
            fp.write(html_str)

        if pdfkit is None:
            logger.warning("Install pdfkit to export report as pdf (https://pypi.org/project/pdfkit/)")
        else:
            pdfkit.from_file(html_path, pdf_path, options = {"enable-local-file-access": None})


    def _make_table(self, rows, column_names=None, row_header=False):
        row_header_class = "row-header" if row_header else "no-row-header"

        if column_names:
            header_cols = [f"<th>{item}</th>" for item in column_names]
            header_html = "\n".join(["<tr>"] + header_cols + ["</tr>"])
        else:
            header_html = ""

        rows_html = []
        for row in rows:
            cols = [f"<td>{item}</td>" for item in row]
            rows_html.append("\n".join(["<tr>"] + cols + ["</tr>"]))
        content_html = "\n".join(rows_html)

        return f"<table class='{row_header_class}'>\n{header_html}\n{content_html}\n</table>"

    def _read_stats_file(self, filename):
        file_path = os.path.join(self.output_path, filename)
        with io.open_rt(file_path) as fin:
            return io.json_load(fin)

    def _make_block(self, content_list):
        content = "\n".join(content_list)
        return f"<div class='block'>\n{content}\n</div>"

    def _make_section(self, title):
        return f"<h2>{title}</h2>"

    def _make_subsection(self, title):
        return f"<h3>{title}</h3>"

    def _make_centered_image(self, image_path, height):
        return f"<div class='centered-image'><img src={quote(image_path)} height='{height}'/></div>"

    def make_title(self):
        version = _git_version()
        return f"<h1>OpenSfM Quality Report</h1>\n<div id='subtitle'>Processed with OpenSfM {version}</div>"

    def make_dataset_summary(self):
        content = [self._make_section("Dataset Summary")]

        rows = [
            ["Dataset", self.dataset_name],
            ["Date", self.stats["processing_statistics"]["date"]],
            [
                "Area Covered",
                f"{self.stats['processing_statistics']['area']/1e6:.6f} km²",
            ],
            [
                "Processing Time",
                f"{self.stats['processing_statistics']['steps_times']['Total Time']:.2f} seconds",
            ],
        ]
        content.append(self._make_table(rows, row_header=True))
        return self._make_block(content)

    def _has_meaningful_gcp(self):
        return (
            self.stats["reconstruction_statistics"]["has_gcp"]
            and "average_error" in self.stats["gcp_errors"]
        )

    def make_processing_summary(self):
        content = [self._make_section("Processing Summary")]

        rec_shots, init_shots = (
            self.stats["reconstruction_statistics"]["reconstructed_shots_count"],
            self.stats["reconstruction_statistics"]["initial_shots_count"],
        )
        rec_points, init_points = (
            self.stats["reconstruction_statistics"]["reconstructed_points_count"],
            self.stats["reconstruction_statistics"]["initial_points_count"],
        )

        geo_string = []
        if self.stats["reconstruction_statistics"]["has_gps"]:
            geo_string.append("GPS")
        if self._has_meaningful_gcp():
            geo_string.append("GCP")

        rows = [
            [
                "Reconstructed Images",
                f"{rec_shots} over {init_shots} shots ({rec_shots/init_shots*100:.1f}%)",
            ],
            [
                "Reconstructed Points",
                f"{rec_points} over {init_points} points ({rec_points/init_points*100:.1f}%)",
            ],
            [
                "Reconstructed Components",
                f"{self.stats['reconstruction_statistics']['components']} component",
            ],
            [
                "Detected Features",
                f"{self.stats['features_statistics']['detected_features']['median']} features",
            ],
            [
                "Reconstructed Features",
                f"{self.stats['features_statistics']['reconstructed_features']['median']} features",
            ],
            ["Geographic Reference", " and ".join(geo_string)],
        ]

        row_gps_gcp = [" / ".join(geo_string) + " errors"]
        geo_errors = []
        if self.stats["reconstruction_statistics"]["has_gps"]:
            geo_errors.append(f"{self.stats['gps_errors']['average_error']:.2f}")
        if self._has_meaningful_gcp():
            geo_errors.append(f"{self.stats['gcp_errors']['average_error']:.2f}")
        row_gps_gcp.append(" / ".join(geo_errors) + " meters")
        rows.append(row_gps_gcp)

        content.append(self._make_table(rows, row_header=True))

        topview_height = 600
        topview_grids = [
            f for f in os.listdir(self.output_path) if f.startswith("topview")
        ]
        content.append(self._make_centered_image(topview_grids[0], topview_height))

        return self._make_block(content)

    def make_processing_time_details(self):
        content = [self._make_section("Processing Time Details")]

        column_names = [f"{i} (s)" for i in self.stats["processing_statistics"]["steps_times"].keys()]
        formatted_floats = []
        for v in self.stats["processing_statistics"]["steps_times"].values():
            formatted_floats.append(f"{v:.2f}")
        rows = [formatted_floats]
        content.append(self._make_table(rows, column_names=column_names))
        return self._make_block(content)

    def make_gps_details(self):
        content = [self._make_section("GPS/GCP Errors Details")]

        # GPS
        for error_type in ["gps", "gcp"]:
            rows = []
            column_names = [error_type.upper(), "Mean", "Sigma", "RMS Error", "Median"]
            if "average_error" not in self.stats[error_type + "_errors"]:
                continue
            for comp in ["x", "y", "z"]:
                row = [comp.upper() + " Error (meters)"]
                row.append(f"{self.stats[error_type + '_errors']['mean'][comp]:.3f}")
                row.append(f"{self.stats[error_type +'_errors']['std'][comp]:.3f}")
                row.append(f"{self.stats[error_type +'_errors']['error'][comp]:.3f}")
                row.append(f"{self.stats[error_type +'_errors']['mad'][comp]:.3f}")
                rows.append(row)

            rows.append(
                [
                    "Total (meters)",
                    "",
                    "",
                    f"{self.stats[error_type +'_errors']['average_error']:.3f}",
                    f"{self.stats[error_type +'_errors']['median_error']:.3f}",
                ]
            )
            content.append(self._make_table(rows, column_names=column_names))
        return self._make_block(content)


    def make_features_details(self):
        content = [self._make_section("Features Details")]

        heatmap_height = 400
        heatmaps = [f for f in os.listdir(self.output_path) if f.startswith("heatmap")]
        content.append(self._make_centered_image(heatmaps[0], heatmap_height))
        if len(heatmaps) > 1:
            logger.warning("Please implement multi-model display")

        column_names = ["", "Min.", "Max.", "Mean", "Median"]
        rows = []
        for comp in ["detected_features", "reconstructed_features"]:
            row = [comp.replace("_", " ").replace("features", "").capitalize()]
            for t in column_names[1:]:
                row.append(
                    f"{self.stats['features_statistics'][comp][t.replace('.', '').lower()]:.0f}"
                )
            rows.append(row)
        content.append(self._make_table(rows, column_names=column_names))

        return self._make_block(content)

    def make_reconstruction_details(self):
        content = [self._make_section("Reconstruction Details")]
        rows = [
            [
                "Average reprojection Error",
                f"{self.stats['reconstruction_statistics']['reprojection_error']:.2f} pixels",
            ],
            [
                "Average Track Length",
                f"{self.stats['reconstruction_statistics']['average_track_length']:.2f} images",
            ],
            [
                "Average Track Length (> 2)",
                f"{self.stats['reconstruction_statistics']['average_track_length_over_two']:.2f} images",
            ],
        ]
        content.append(self._make_table(rows, row_header=True))
        return self._make_block(content)

    def make_camera_models_details(self):
        content = [self._make_section("Camera Models Details")]

        for camera, params in self.stats["camera_errors"].items():
            residual_grids = [
                f
                for f in os.listdir(self.output_path)
                if f.startswith("residuals_" + str(camera.replace("/", "_")))
            ]
            if not residual_grids:
                continue

            initial = params["initial_values"]
            optimized = params["optimized_values"]
            names = [""] + list(initial.keys())

            rows = []
            rows.append(["Initial"] + [f"{x:.4f}" for x in initial.values()])
            rows.append(["Optimized"] + [f"{x:.4f}" for x in optimized.values()])

            content.append(self._make_subsection(camera))
            content.append(self._make_table(rows, column_names=names))

            residual_grid_height = 400
            content.append(self._make_centered_image(residual_grids[0], residual_grid_height))

        return self._make_block(content)

    def make_tracks_details(self):
        content = [self._make_section("Tracks Details")]
        matchgraph_height = 400
        matchgraph = [
            f for f in os.listdir(self.output_path) if f.startswith("matchgraph")
        ]
        content.append(self._make_centered_image(matchgraph[0], matchgraph_height))

        histogram = self.stats["reconstruction_statistics"]["histogram_track_length"]
        start_length, end_length = 2, 10
        row_length = ["Length"]
        for length, count in sorted(histogram.items(), key=lambda x: int(x[0])):
            if int(length) < start_length or int(length) > end_length:
                continue
            row_length.append(length)
        row_count = ["Count"]
        for length, count in sorted(histogram.items(), key=lambda x: int(x[0])):
            if int(length) < start_length or int(length) > end_length:
                continue
            row_count.append(f"{count}")

        content.append(self._make_table([row_length, row_count], row_header=True))

        return self._make_block(content)

    def generate_report(self):
        parts = [
            self.make_title(),
            self.make_dataset_summary(),
            self.make_processing_summary(),
            self.make_features_details(),
            self.make_reconstruction_details(),
            self.make_tracks_details(),
            self.make_camera_models_details(),
            self.make_gps_details(),
            self.make_processing_time_details(),
        ]
        body = "\n".join(parts)
        self.html = f"""
        <html>
          <head>
            <meta charset='utf-8'>
            <style>
              {css_style}
            </style>
          </head>
          <body>
            {body}
          </body>
        </html>"""
