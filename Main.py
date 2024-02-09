# Name: Main.py
# Auth: Kareem T
# Date: 2/8/24
# Desc: Main user program that runs experiments
from os import listdir, getcwd
import Experiment as Exp

def run_experiments():
    data_dir = getcwd() + '/data/'
    grand = Exp.Grand_Experiment()
    for i, file in enumerate(listdir(data_dir)):
        print(f'Experiment {i}: {file}')
        file_path = data_dir + file
        Exp.Experiment.run(file_path)
        grand.add_data(file_path)
    print(f'Final Experiment: Grand Experiment')
    grand.run()

def main():
    run_experiments()

if __name__ == '__main__':
    main()