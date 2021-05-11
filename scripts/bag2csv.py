"""
File that reads a single rosbag or all rosbags in a directory and 
creats csv files for choosen topics in the bag.

Usage: 
    for a single bagfile
    python bag2csv.py <bag_path> <topic1 topic2 topic3...>
    
    for all bags in a directory
    python bag2csv.py -d <bag_dir> <topic1 topic2 topic3...>
   


"""

import sys
from typing import List
from pathlib import Path
from bagpy import bagreader


def bag2csv(bag_path: Path, topics: List[str]) -> None:

    bag = bagreader(str(bag_path.resolve()))

    # convert topics to csv files
    # csv files will be stored in a folder with the name of the bag
    for topic in topics:
        bag.message_by_topic(topic)


def convert_bags_in_dir(directory: Path, topics: List[str]) -> None:

    for element in directory.iterdir():
        if element.is_file() and element.suffix == ".bag":
            bag2csv(element, topics)


if __name__ == "__main__":

    opts = [opt for opt in sys.argv[1:] if opt.startswith("-")]
    args = [arg for arg in sys.argv[1:] if not arg.startswith("-")]

    if "-d" in opts:
        bag_dir = Path(args[0])
        topics = args[1:]
        convert_bags_in_dir(bag_dir, topics)

    else:
        bag_path = Path(args[0])
        topics = args[1:]
        convert_bags_in_dir(bag_path, topics)
