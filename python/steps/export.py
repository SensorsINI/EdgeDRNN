__author__ = "Chang Gao"
__copyright__ = "Copyright @ Chang Gao"
__credits__ = ["Chang Gao"]
__license__ = "Private"
__version__ = "0.0.1"
__maintainer__ = "Chang Gao"
__email__ = "gaochangw@outlook.com"
__status__ = "Prototype"

from project import Project
from modules.edgedrnn import EdgeDRNN

def main(proj: Project):
    edgedrnn = EdgeDRNN(proj)
    edgedrnn.collect_params()
    edgedrnn.collect_test_data()
    edgedrnn.gen_lut()
    print("EdgeDRNN Export is Done......")