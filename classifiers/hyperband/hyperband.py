import numpy as np
import json 

from random import random
from math import log, ceil
from time import time, ctime

class Hyperband:
        
        def __init__(self, training_data, get_hyp_config, run_hyp_config):
                """
                Inputs: get_hyp_config(n): 
                        returns a set of n i.i.d. samples from a distribution over the hyperparameter space
                        run_hyp_config(t, r):
                        accepts hyperparameter configuration (t) and resource allocation (r) as input, returns
                        validation loss after training configuration for the allocated resources
                """
                self.training_data = training_data
                self.get_params = get_hyp_config
                self.try_params = run_hyp_config
               
                # Authors: We suggest that max_iter be set to the number of iterations you would use if your supervisor gave
                # you a hyperparameter configuration and asked for a model back
                # Iteration defined as a unit of computation

                self.max_iter = 81      # maximum iterations per configuration
                self.eta = 3            # defines configuration downsampling rate (default = 3)

                self.logeta = lambda x: log(x) / log(self.eta)
                self.s_max = int(self.logeta(self.max_iter))
                self.B = (self.s_max + 1) * self.max_iter

                self.results = []       # list of dicts
                self.counter = 0
                self.best_auc = 0
                self.best_counter = -1
                

        # can be called multiple times
        # skip baseline uniform allocation (random search at bracket s = 0) if desired
        def run(self, skip_random_search = False):
                
                for s in reversed(range(self.s_max + 1)):
                        
                        # initial number of configurations
                        n = int( ceil( self.B / self.max_iter / ( s + 1 ) * self.eta ** s ))    
                        
                        # initial number of iterations per config - (resource allocation)
                        r = self.max_iter * self.eta ** (-s)            

                        # n random configurations
                        T = [ self.get_params() for i in range(n)] 
                        
                        for i in range((s + 1) - int(skip_random_search)):
                                
                                # Run each of the n configs for <iterations> 
                                # and keep best (n_configs / eta) configurations
                                
                                n_configs = n * self.eta**(-i)
                                n_iterations = r * self.eta**i
                                
                                print("\n*** {} configurations x {:.1f} iterations each".format( 
                                        n_configs, n_iterations))
                                
                                val_auc = []
                                early_stops = []
                                
                                # Inner loop over hyperparameter configurations (t), perform successiveHalving
                                for t in T:
                                        
                                        self.counter += 1
                                        print("\nrun {} | {} | Leader: {:.4f} (run {})\n".format( 
                                                self.counter, ctime(), self.best_auc, self.best_counter))
                                        
                                        start_time = time()
                                        
                                        result = self.try_params(self.training_data, hyp_params = t, n_iterations = n_iterations)
                                                
                                        assert(type( result ) == dict)
                                        assert('auc' in result)
                                        
                                        seconds = int( round( time() - start_time ))
                                        print("Training ended. Elapsed: \n{} s.".format(seconds))
                                        
                                        auc = result['auc']   
                                        val_auc.append(auc)
                                        
                                        early_stop = result.get( 'early_stop', False )
                                        early_stops.append( early_stop )
                                        
                                        # keeping track of the best result so far (for display only)
                                        if auc > self.best_auc:
                                                self.best_auc = auc
                                                self.best_counter = self.counter
                                        
                                        result['counter'] = self.counter
                                        result['seconds'] = seconds
                                        result['params'] = t
                                        result['iterations'] = n_iterations
                                        
                                        self.results.append( result )

                                        # Write results as we go
                                        with open('hparams_running.json', 'w') as f:
                                            json.dump(x, f)

                                
                                # select a number of best configurations for the next loop
                                # filter out early stops, if any
                                indices = np.argsort(val_auc)
                                T = [ T[i] for i in indices if not early_stops[i]]
                                T = T[ 0:int( n_configs / self.eta )]
                
                return self.results
        

