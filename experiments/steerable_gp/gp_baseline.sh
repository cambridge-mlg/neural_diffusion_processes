python experiments/steerable_gp/gp_baseline.py -m data.num_samples_test=10_000 eval.batch_size=100 data_kernel=curlfree,divfree,rbfvec logger=all seed=0,1 name=gp_baseline