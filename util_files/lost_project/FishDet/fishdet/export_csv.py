from lost.pyapi import script
import os
import pandas as pd
import datetime
ENVS = ['lost']

class ExportCsv(script.Script):
    '''This Script writes annos from task to csv in annos_out dir.
    '''
    def main(self):
        df = self.inp.to_df()
        anno_path = "home/lost/FishDet/annos_out/"
        filename = "annos_"+datetime.datetime.today().strftime("%y%m%d%H%M")
        csv_path = anno_path+filename
        df.to_csv(path_or_buf=csv_path,
                      sep=',',
                      header=True,
                      index=False)
if __name__ == "__main__":
    my_script = ExportCsv()
