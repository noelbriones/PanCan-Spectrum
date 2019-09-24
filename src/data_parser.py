#!/usr/bin/env python
import pymzml
from pymzml.run import Reader


def parse(dataset_location):
    example_file = dataset_location
    run = pymzml.run.Reader(example_file)
    count = Reader(example_file).get_spectrum_count()
    print(count)

    spectrum_list = [[0 for x in range(3)] for y in range(10)]
    n = 0
    for mass_spec in run:
        mz = list(mass_spec.mz)
        i = list(mass_spec.i)
        rt = list(mass_spec.scan_time[:len(mass_spec.scan_time) - 1])
        spectrum_list[n][0] = mz
        spectrum_list[n][1] = i

        spectrum_list[n][2] = rt
        n += 1
        if n > 9:
            break
    return spectrum_list
