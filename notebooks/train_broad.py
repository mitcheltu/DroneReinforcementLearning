"""Run the broad non-solid curriculum without replacing deployed models."""
import argparse
from pathlib import Path
from notebooks import train_maneuver as runner
from notebooks import broad_maneuver as curriculum


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--source',default='runs/agile-curriculum-006/level-9-round-3/model.zip')
    for name,default in [('episodes',24),('epochs',40),('rounds',4),('ppo-steps',8192),('cases',12),('final-cases',32)]:
        parser.add_argument('--'+name,type=int,default=default)
    args=vars(parser.parse_args())
    output=Path(args['output'])
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use a new empty output directory')
    # Explicit curriculum injection into the shared fitter/evaluator. Checkpoint
    # architecture and browser-compatible observation contract remain unchanged.
    runner.PROFILES=curriculum.PROFILES
    runner.reset_course=curriculum.reset_course
    runner.expert=curriculum.expert
    # The runner snapshots its core sources; archive the extension after mkdir.
    original_dump=runner.dump
    def dump(path,value):
        if path.name=='configuration.json':
            for name in ('broad_maneuver.py','train_broad.py'):
                (path.parent/name).write_bytes(Path(__file__).with_name(name).read_bytes())
        original_dump(path,value)
    runner.dump=dump
    runner.run(**args,lessons=curriculum.LESSONS,old_levels=[-1]+list(range(13)),
               learning_rate=3e-5,continue_on_failure=True,seed_offset=100000000)


if __name__=='__main__':
    main()
