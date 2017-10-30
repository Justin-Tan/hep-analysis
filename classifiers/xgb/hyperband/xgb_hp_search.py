#!/usr/bin/env python

# XGBoost hyperparameter tuning using HyperBand
# Date: May 2017

import pandas as pd
import sys, time, os
import argparse
import json

import methods.xgb
from hyperband import Hyperband
from pprint import pprint

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_file', help = 'Path to training dataset')
    parser.add_argument('identifier', help = 'training identifier')
    args = parser.parse_args()
    
    print('Loading dataset from: %s with test size 0.05' %(args.data_file))
    # dataDMatrix = methods.xgb.load_data(args.data_file, args.mode, args.channel)
    dataDMatrix = methods.xgb.load_data(args.data_file)

    start = time.time()
    print('Running HyperBand')
    hb = Hyperband(dataDMatrix, methods.xgb.get_hyp_config, methods.xgb.run_hyp_config)
    results = hb.run(skip_random_search = True)

    delta_t = time.time() - start
    output = os.path.join('models', '{}.json'.format(args.identifier))
    
    print("{} Total, Leaderboard:\n".format(len(results)))

    for r in sorted(results, key = lambda x: x['auc'])[:10]:
        print("# auc: {:.2%} | {} s | {:.1f} iterations | run {} ".format( 
                        r['auc'], r['seconds'], r['iterations'], r['counter']))
        pprint(r['params'])
        print

    print('Hyperparameter search complete. Results saved to %s\n' %(output))
    with open(output, 'w') as f:
        json.dump(results, f)
