#Utils to filter files based on extension and 
#basic quality of the file.
from os import path
from json import load
from typing import Dict, List

file_path = path.abspath(__file__)
def load_extension_manage_file():
    """
    Loads the extension managememnt file for filtering..
    """
    return load(open(file_path.split("/filtering_files.py")[0]+r"/filter_extension.json","r"))



class FilterData:
    def __init__(self) -> None:
        self.filter_extension = load_extension_manage_file()
    
    def filter_file_extension(self,datapoint):
        """
        Given a datapoint
        """
        file_name  =  datapoint["file_name"]
        if file_name.split(".")[-1] in self.filter_extension["additive_extensions"]:
            return True
        elif file_name.split(".")[-1] not in self.filter_extension["additive_extensions"]:
            if file_name.split(".")[-1] not in self.filter_extension["deductive_extensions"]:
                return True
        else:
            return False
    def __call__(self,datapoint_list:List[Dict]):
        """
        Complete Filtering Criteria of the datapoint_list(List[Dict])
        """
        check_dict = {}
        filtered_datapoint = []
        for datapoint in datapoint_list:
            check_dict["filter_path_check"] = self.filter_file_extension(datapoint) #should have "file_name" key
            print(list(check_dict.values()))
            if set(list(check_dict.values())) == set([True]) :
                filtered_datapoint.append(datapoint)
            else:
                pass
        return filtered_datapoint

if __name__ == "__main__":
    None
    print(FilterData()([{"file_name":"ast.py"},{"file_name":"ast.json"}]))