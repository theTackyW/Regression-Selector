# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 15:39:07 2022

@author: Jacky
"""

import regression_modules as module
from os import path
import sys

command = sys.argv

if ("-f" in command):
    data = command[command.index("-f") +1]
elif ("-h" in command):
    direct = path.dirname(__file__)
    with open(path.join(direct,"README.md")) as file:
        while(True):
            line = file.readline()
            if(len(line) != 0 and line.find('@help') != -1):
                print(line, end='')
                while(True):
                    line = file.readline()
                    if(len(line) != 0 and line.find('@') != -1):
                        sys.exit("Good luck and have a nice day")
                    print(line, end='')
else:
    sys.exit("Invalid input command, please use '-h' for help ")
poly = module.Polynomial_reg(data)
multi = module.multi_linear_reg(data)
dtree = module.decision_tree_reg(data)
svr = module.svr(data)
randomForest = module.random_forest_reg(data)

poly.report_r2()
multi.report_r2()
dtree.report_r2()
svr.report_r2()
randomForest.report_r2()

