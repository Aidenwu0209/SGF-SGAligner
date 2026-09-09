"""Isolate the existing SGA Torch/CUDA environment from native SGF."""
import argparse
from pathlib import Path
from .semantic_mapping import associate, load_submap, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('source','target','relation-vocab','output'):
        parser.add_argument('--'+flag,type=Path,required=True)
    parser.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    accepted, report = associate(load_submap(args.source),load_submap(args.target),
                                 args.relation_vocab,args.device)
    report['accepted_pairs'] = accepted
    write_json(args.output,report)


if __name__ == '__main__':
    main()
