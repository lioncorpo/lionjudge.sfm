from . import command
from opensfm.actions import mesh


class Command(command.CommandBase):
    name = 'mesh'
    help = "Add delaunay meshes to the reconstruction"

    def run_impl(self, dataset, args):
        mesh.run_dataset(dataset)

    def add_arguments_impl(self, parser):
        pass
