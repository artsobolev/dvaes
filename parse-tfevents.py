import tensorflow as tf

import os
import ujson
import argparse

from collections import defaultdict


def parse_tfevents(path):
    res = defaultdict(list)

    try:
        for event in tf.train.summary_iterator(path):
            for value in event.summary.value:
                res[value.tag].append( (event.step, event.wall_time, value.simple_value) )
    except:
        print "An error has occurred"

    return res


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('tfevents', type=str, help='Path to the tfevents file')
    argparser.add_argument('save_to', type=str, help='Path to save the results')

    args = argparser.parse_args()

    data = parse_tfevents(args.tfevents)

    if not os.path.exists(args.save_to):
        os.mkdir(args.save_to)

    for key, value in data.iteritems():
        key = key.replace('/', '_')
        with open('{}/{}.json'.format(args.save_to, key), 'wb') as fh:
            ujson.dump(value, fh)
