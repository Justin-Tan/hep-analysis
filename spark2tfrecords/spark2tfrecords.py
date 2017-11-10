#!/usr/bin/env python3
"""
 Justin Tan 2017
Reads parquet file, conducts preprocessing, outputs to TfRecords
spark-submit --name spark2tf --jars /home/jtan/gpu/jtan/spark/ecosystem/spark/spark-tensorflow-connector/target/spark-tensorflow-connector-1.0-SNAPSHOT.jar,/home/jtan/gpu/jtan/spark/ecosystem/spark/spark-tensorflow-connector/target/lib/tensorflow-hadoop-1.0-06262017-SNAPSHOT-shaded-protobuf.jar --properties-file default.cfg <driver.py> <path/to/parquet>

"""
import numpy as np
import pandas as pd
import sys, glob, os, multiprocessing
import argparse

import findspark
findspark.init('/home/jtan/anaconda3/envs/spark/lib/python3.6/site-packages/pyspark')
import pyspark

def df_preprocess(df_pqt):
    from pyspark.ml.stat import Correlation
    from pyspark.ml.feature import VectorAssembler
    from pyspark.ml.feature import StandardScaler
    subexclude = ['weight', 'index']
    excludecols = [col for col in df_pqt.columns for subcol in subexclude if subcol in col]
    df = df_pqt.drop(*excludecols)

    print('Correlation between Delta E, Mbc: {}'.format(df.corr('mbc','deltae')))

    specialCols = ['mbc', 'deltae', 'labels']
    featureCols = [col for col in df.columns if 'label' not in col]
    assembler = VectorAssembler(inputCols=featureCols, outputCol='raw_features')
    df_features = assembler.transform(df).select('raw_features') #, 'labels')

    # Compute correlation matrix, drop features correlated to fit variables
    corr = Correlation.corr(df_features, 'raw_features', 'pearson').collect()[0][0]
    df_corr = pd.DataFrame(corr.toArray(), index=featureCols, columns=featureCols)
    corr2fit = correlated2fit(df_corr)
    df = df.drop(*corr2fit)

    trainCols = [col for col in df.columns if col not in specialCols]
    assembler = VectorAssembler(inputCols=trainCols, outputCol='trainFeatures')
    df = assembler.transform(df)

    # Train-test split
    df_train, df_test = df.randomSplit([0.9, 0.1], seed=42)

    # Normalization - unit variance, zero mean
    # Here the output column is a vector of scaled features
    scaler = StandardScaler(inputCol='trainFeatures', outputCol='features',
                            withMean=True, withStd=True)
    # Compute summary statistics by fitting the StandardScaler on training set
    scalerModel = scaler.fit(df_train)
    # Use training statistics to normalize train+test dataset, save as an extra column
    scaled_df_train = scalerModel.transform(df_train)
    scaled_df_test = scalerModel.transform(df_test)
    sc_train = scaled_df_train.select(*specialCols, 'features')
    sc_test = scaled_df_test.select(*specialCols, 'features')

    return sc_train, sc_test


"""
    sc_df_train = unpack_vectorCol(scaled_df_train, trainCols, specialCols)
    sc_df_test = unpack_vectorCol(scaled_df_test, trainCols, specialCols)
    sc_df_train.write.parquet('/home/jtan/gpu/jtan/torch/sc_train.parquet', compression='snappy')
    sc_df_test.write.parquet('/home/jtan/gpu/jtan/torch/sc_test.parquet', compression='snappy')

    sc_test_readback = spark.read.parquet('/home/jtan/gpu/jtan/torch/sc_test.parquet')
    test=sc_test_readback.toPandas()
    test.describe()
"""
def correlated2fit(corr, corr_threshold = 0.95):
    # List features correlated to fit variables B_mbc and B_deltae
    fit_variables = [branch for branch in corr.columns if 'mbc' in branch or 'deltae' in branch]
    corr2fit = []
    for variable in fit_variables:
        if variable not in fit_variables:
            corr2fit += corr.loc[lambda df: df[variable] > corr_threshold].index.tolist()
    corr2fit = list(set(corr2fit))# + fit_variables))

    print('Removing variables correlated to mbc, deltaE:')
    if not corr2fit:
        print('(None correlated)')
    [print('## {}'.format(variable)) for variable in corr2fit]
    return corr2fit

def get_redundant_pairs(cols):
    '''Get diagonal and lower triangular pairs of correlation matrix'''
    pairs_to_drop = set()
    for i in range(len(cols)):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop

def get_top_abs_correlations(df_corr, n=50):
    au_corr = df_corr.abs().unstack()
    labels_to_drop = get_redundant_pairs(df_corr.columns)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[0:n]

def rm_correlated(df_corr):
    d_corr = get_top_abs_correlations(df_corr, n = 128)
    corr_pairs = [tupl for tupl in list(d_corr[d_corr > 0.99999].index)]
    top001_left = [tupl[0] for tupl in corr_pairs]
    top001_right = [tupl[1] for tupl in corr_pairs]

    [print(feature) for feature in list(set(top001_right)) if feature in list(set(top001_left))]
    to_remove = [feature for feature in list(set(top001_right)) if feature not in top001_left]

    return to_remove

def unpack_vectorCol(df, trainCols, specialCols, col2unpack='scaledFeatures'):
    from pyspark.sql.functions import udf, col
    from pyspark.sql.types import ArrayType, DoubleType

    def to_array(col):
        def to_array_(v):
            return v.toArray().tolist()
        return udf(to_array_, ArrayType(DoubleType()))(col)

    unpacked_df = (df.withColumn("sf", to_array(col(col2unpack)))
                     .select([col("sf")[i].alias(trainCols[i]) for i in range(len(trainCols))]+specialCols))

    return unpacked_df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('parquet', help='Path to parquet file to read')
    #parser.add_argument('out', required=True, help='Output TFRecords directory')
    args = parser.parse_args()

    print('Running on {}-core machine, set local[*] appropriately'.format(multiprocessing.cpu_count()))
    base = os.path.splitext(os.path.basename(args.parquet))[0]

    conf = pyspark.SparkConf().setAll([('spark.app.name', 'Spark2TfRecords'), ('spark.master', 'local[*]'),
                                       ('spark.executor.memory', '16g'), ('spark.driver.memory','16g')])

    spark = pyspark.sql.SparkSession.builder \
    .config(conf=conf) \
    .getOrCreate()

    df_pqt = spark.read.parquet(args.parquet)
    # df_pqt.write.format('tfrecords').save("example.tfrecords")
    print(df_pqt.columns)

    scaled_train, scaled_test = df_preprocess(df_pqt)
    # scaled_train.write.parquet('sc_{}_train.parquet'.format(base), compression='snappy')
    scaled_test.write.parquet('sc_{}_test.parquet'.format(base), compression='snappy')

    # Write to TFRecords format
    scaled_test.write.format('tfrecords').save("example_test.tfrecords")

if __name__ == "__main__":
    main()
