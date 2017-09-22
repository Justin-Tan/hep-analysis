#!/usr/bin/env python3
"""
Converts hdf5 in table format to parquet file
"""
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import argparse
import time

def convert_hdf5_to_parquet(h5_file, parquet_file, chunksize=100000):

    stream = pd.read_hdf(h5_file, chunksize=chunksize)
    print('Starting conversion ...')

    for i, chunk in enumerate(stream):
        print("### Chunk {}".format(i))

        if i == 0:
            # Infer schema and open parquet file on first chunk
            parquet_schema = pa.Table.from_pandas(df=chunk).schema
            parquet_writer = pq.ParquetWriter(parquet_file, parquet_schema, compression='snappy')

        table = pa.Table.from_pandas(chunk, schema=parquet_schema)
        parquet_writer.write_table(table)

    parquet_writer.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('h5_file', help='Path to hdf5 file in table format')
    parser.add_argument('--out', required=True, help='Output parquet file')
    args = parser.parse_args()

    t0 = time.time()
    convert_hdf5_to_parquet(args.h5_file, args.out)
    print('Conversion time: {}'.format(time.time() - t0))

if __name__ == "__main__":
    main()
