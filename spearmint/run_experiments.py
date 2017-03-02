import os
import sys
import subprocess
import shlex
import argparse
from spearmint.utils.parsing import repeat_output_dir

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('expdir', type=str, nargs='+')
    parser.add_argument('--repeat_start', type=int, default=0)
    parser.add_argument('--repeat', type=int, default=1)
    # parser.add_argument('--pause_on', type=bool, default=True)
    parser.add_argument('--runs_at_once', type=int, default=1)
    args = parser.parse_args()

    # unpack the arguments
    repeat = args.repeat
    repeat_start = args.repeat_start
    # pause_on = args.pause_on
    exp_dirs = args.expdir
    runs_at_once = args.runs_at_once

    for expt_dir in exp_dirs:
        for i in xrange(repeat_start, repeat):
            print 'Running experiment %d/%d.' % (i+1, repeat)

            # Create output directory if needed and open output file
            output_dir = repeat_output_dir(expt_dir, i)

            if not os.path.isdir(output_dir):
                os.mkdir(output_dir)
            output_filename = os.path.join(output_dir, "main.log")
            output_file = open(output_filename, 'w')

            p = subprocess.Popen(shlex.split("python /Users/jamesrequeima/Dropbox/PhD/Code/Spearmint/spearmint/main.py %s "
                                             "--repeat=%d" % (expt_dir, i)),
                stdout=output_file, stderr=output_file)
            if (i+1)%runs_at_once == 0:
                p.wait()


if __name__ == '__main__':
    main()