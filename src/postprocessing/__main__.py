"""One entry point for post-simulation analysis; protocols load only when selected."""
import argparse
import runpy
import sys


COMMANDS = {
    'inventory': ('inventory', 'Verify production sources and freeze extraction inputs'),
    'descriptors': ('protocols.search', 'Prepare/extract structural labels and inspect progress'),
    'targets': ('targets', 'Prepare rich TDA and bond-order spatial-training targets on CPU'),
    'spatial-dataset': ('spatial_dataset', 'Extend the spatial fitting set with Ni and prepare compact targets'),
    'spatial-enrich': ('spatial_enrichment', 'Add verified Ni crystal interiors and interfaces to spatial training'),
    'spatial-relaxed': ('relaxed_spatial', 'Prepare paired relaxed structural targets and queue Small eSEN'),
    'target-benchmark': ('target_benchmark', 'Audit and time rich spatial targets against an explicit reference'),
    'gpu-benchmark': ('gpu_benchmark', 'Audit experimental CUDA descriptors on pinned real frames'),
    'gpu-extract': ('gpu_extract', 'Stream opt-in GPU extraction to a separate external cache'),
    'cache': ('cache', 'Copy existing labels to selective caches without rerunning descriptors'),
    'search': ('search', 'Track nucleus families, collect results, or submit CPU workers'),
    'dataset': ('datasets.births', 'Refresh, inspect or verify the cumulative birth catalogue'),
    'labels': ('datasets.labels', 'Publish and verify observation-label supplements'),
    'histories': ('histories.al', 'Extract birth-centred histories and precursor graphs'),
    'publish': ('histories.publish', 'Queue, collect and publish birth-history datasets'),
    'view': ('visualization.births', 'Render birth movies, plots and the dataset index'),
    'export': ('visualization.export', 'Export portable interactive birth galleries'),
    'ni-search': ('protocols.ni', 'Analyse or combine completed Ni campaign sources'),
    'ni-gpu-search': ('protocols.ni_gpu', 'Queue GPU Ni descriptors and unchanged CPU lineage tracking'),
    'al-gpu-search': ('protocols.al_gpu', 'Analyse verified Al additions on GPU and publish the completed birth catalogue'),
    'ni-histories': ('histories.ni', 'Extract the declared Ni birth-history examples'),
    'ni-preview': ('visualization.ni', 'Plot Ni monitoring and retrospective local order'),
    'crops': ('visualization.crops', 'Export saved full-cell event neighbourhoods'),
    'crop-windows': ('visualization.crop_windows', 'Export temporal windows of event crops'),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='Commands:\n' + '\n'.join(f'  {name:16} {info[1]}' for name, info in COMMANDS.items()) +
        '\n\nUse COMMAND --help for its stages and arguments. No work starts without a command.')
    parser.add_argument('command', choices=COMMANDS)
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    module = 'src.postprocessing.' + COMMANDS[args.command][0]
    sys.argv = [f'{parser.prog} {args.command}', *args.arguments]
    runpy.run_module(module, run_name='__main__')


if __name__ == '__main__':
    main()
