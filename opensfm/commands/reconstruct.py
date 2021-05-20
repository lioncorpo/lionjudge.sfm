import logging
import time

from opensfm import dataset
from opensfm import io
from opensfm import reconstruction
from opensfm import pymap
logger = logging.getLogger(__name__)


class Command:
    name = 'reconstruct'
    help = "Compute the reconstruction"

    def add_arguments(self, parser):
        parser.add_argument('dataset', help='dataset to process')

    def run(self, args):
        start = time.time()
        data = dataset.DataSet(args.dataset)
        tracks_manager = data.load_tracks_manager()
        report, reconstructions = reconstruction.\
            incremental_reconstruction(data, tracks_manager)
        end = time.time()
        with open(data.profile_log(), 'a') as fout:
            fout.write('reconstruct: {0}\n'.format(end - start))
        data.save_reconstruction(reconstructions)
        # pymap.MapIO.save_map(reconstructions[0].map, data._reconstruction_file(None))
        data.save_report(io.json_dumps(report), 'reconstruction.json')
