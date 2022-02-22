__author__ = "Chang Gao"
__copyright__ = "Copyright @ Chang Gao"
__credits__ = ["Chang Gao"]
__license__ = "Private"
__version__ = "0.0.1"
__maintainer__ = "Chang Gao"
__email__ = "gaochangw@outlook.com"
__status__ = "Prototype"

from project import Project
from steps import train, export

if __name__ == '__main__':
    # Declare a Project Object
    proj = Project()

    # Step 0 - Pretrain on GRU
    if proj.step == 'pretrain':
        print("####################################################################################################\n"
              "# Step 0: Pretrain                                                                                  \n"
              "####################################################################################################")
        train.main(proj)
        proj.step_in()

    # Step 1 - Retrain on DeltaGRU
    if proj.step == 'retrain':
        print(
            "####################################################################################################\n"
            "# Step 1: Retrain                                                                                   \n"
            "####################################################################################################")
        train.main(proj)
        proj.step_in()

    # Step 2 - Export to EdgeDRNN
    if proj.step == 'export':
        print(
            "####################################################################################################\n"
            "# Step 2: Export                                                                                   \n"
            "####################################################################################################")
        export.main(proj)